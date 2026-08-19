"""Whole-corpus run over the real 24 documents.

Guards the properties the retrieval layer will rely on and that unit tests on
single documents cannot see: every manifest document is indexed, ids are unique
across the corpus, and a second run is a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import IngestionSettings
from app.ingestion.pipeline import IngestionPipeline, discover_documents


@pytest.fixture(scope="module")
def manifest(request) -> list[dict]:
    path = Path(request.config.rootpath).parents[0] / "data" / "corpus_manifest.json"
    if not path.is_file():
        pytest.skip(f"manifest not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def full_run(corpus: Path):
    from app.ingestion.embedder import DeterministicEmbedder
    from app.ingestion.indexer import InMemoryIndexer

    embedder = DeterministicEmbedder(dimensions=16)
    indexer = InMemoryIndexer()
    pipeline = IngestionPipeline(
        settings=IngestionSettings(
            CORPUS_ROOT=corpus, SOURCE_URI_BASE="https://acct.blob.core.windows.net/corpus/raw"
        ),
        embedder=embedder,
        indexer=indexer,
    )
    report = pipeline.run(discover_documents(corpus))
    return pipeline, report, indexer


def test_whole_corpus_ingests_without_failures(full_run) -> None:
    _, report, _ = full_run

    assert report.status == "succeeded"
    assert report.docs_failed == 0
    assert report.docs_indexed == report.docs_total


def test_every_manifest_document_is_indexed(full_run, manifest) -> None:
    _, _, indexer = full_run
    indexed = {d["document_id"] for d in indexer.documents.values()}

    expected = {entry["document_id"] for entry in manifest}
    assert expected == indexed


def test_format_counts_match_the_manifest(full_run, manifest) -> None:
    _, report, _ = full_run

    from collections import Counter

    actual = Counter(r.document_type for r in report.results)
    expected = Counter(entry["format"].lower().replace("md", "markdown") for entry in manifest)
    assert actual == expected


def test_document_names_match_the_manifest_titles(full_run, manifest) -> None:
    _, _, indexer = full_run
    names = {d["document_id"]: d["document_name"] for d in indexer.documents.values()}

    for entry in manifest:
        assert names[entry["document_id"]] == entry["title"]


def test_chunk_ids_are_unique_across_the_corpus(full_run) -> None:
    _, report, indexer = full_run

    assert len(indexer.documents) == report.chunks_written


def test_every_chunk_has_content_and_a_citable_section(full_run) -> None:
    _, _, indexer = full_run

    for document in indexer.documents.values():
        assert document["content"].strip()
        assert document["section"] and document["section_path"]
        assert document["source_uri"].startswith("https://acct.blob.core.windows.net/corpus/raw/")
        assert document["token_count"] > 0


def test_pdf_chunks_carry_page_numbers(full_run) -> None:
    _, _, indexer = full_run
    pdf_chunks = [d for d in indexer.documents.values() if d["document_type"] == "pdf"]

    assert pdf_chunks
    assert all(isinstance(d["page_number"], int) for d in pdf_chunks)


def test_rerunning_the_whole_corpus_is_a_no_op(full_run, corpus: Path) -> None:
    pipeline, first, indexer = full_run
    before = set(indexer.documents)

    second = pipeline.run(discover_documents(corpus))

    assert second.docs_skipped == first.docs_total
    assert second.docs_indexed == 0
    assert second.chunks_written == 0
    assert set(indexer.documents) == before
