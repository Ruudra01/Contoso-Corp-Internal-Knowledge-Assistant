"""End-to-end pipeline behaviour: every format, idempotency, re-index safety,
and per-document error isolation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.core.config import ChunkingSettings, IngestionSettings
from app.core.errors import IndexingError, IngestionError
from app.ingestion.embedder import DeterministicEmbedder
from app.ingestion.indexer import InMemoryIndexer
from app.ingestion.pipeline import (
    IngestionPipeline,
    Outcome,
    discover_documents,
)
from tests.conftest import SAMPLES

ALL_TYPES = ["pdf", "docx", "html", "markdown"]


@pytest.fixture
def pipeline(embedder, indexer, tmp_path) -> IngestionPipeline:
    settings = IngestionSettings(CORPUS_ROOT=tmp_path, SOURCE_URI_BASE="https://blob/corpus/raw")
    return IngestionPipeline(settings=settings, embedder=embedder, indexer=indexer)


@pytest.fixture
def one_of_each(tmp_path: Path, corpus: Path) -> list[Path]:
    """A four-document corpus, one per supported format."""
    staged = []
    for path in SAMPLES.values():
        if not path.is_file():
            pytest.skip(f"sample missing: {path}")
        target = tmp_path / path.name
        shutil.copy(path, target)
        staged.append(target)
    return sorted(staged)


# -- discovery -----------------------------------------------------------------


def test_discovery_finds_every_supported_format(one_of_each: list[Path], tmp_path: Path) -> None:
    assert discover_documents(tmp_path) == one_of_each


def test_discovery_ignores_unsupported_and_repository_files(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "README.md").write_text("# repo docs", encoding="utf-8")
    (tmp_path / ".DS_Store").write_bytes(b"\x00")
    (tmp_path / "cnt-hr-900_real.md").write_text("# Real\n\nBody.", encoding="utf-8")

    assert [p.name for p in discover_documents(tmp_path)] == ["cnt-hr-900_real.md"]


def test_discovery_recurses_into_subdirectories(tmp_path: Path) -> None:
    nested = tmp_path / "pdf" / "deep"
    nested.mkdir(parents=True)
    (nested / "cnt-hr-900_a.md").write_text("# A\n\nBody.", encoding="utf-8")

    assert len(discover_documents(tmp_path)) == 1


def test_discovery_rejects_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="does not exist"):
        discover_documents(tmp_path / "nope")


# -- happy path ----------------------------------------------------------------


def test_all_four_formats_ingest_successfully(pipeline, one_of_each, indexer) -> None:
    report = pipeline.run(one_of_each)

    assert report.status == "succeeded"
    assert report.docs_total == 4
    assert report.docs_indexed == 4
    assert report.docs_failed == 0
    assert report.chunks_written == len(indexer.documents) > 0
    assert {r.document_type for r in report.results} == {"pdf", "docx", "html", "markdown"}


def test_indexed_documents_carry_metadata_and_vectors(pipeline, one_of_each, indexer, embedder) -> None:
    pipeline.run(one_of_each)

    for document in indexer.documents.values():
        assert document["document_id"] and document["document_name"]
        assert document["document_type"] in {"pdf", "docx", "html", "markdown"}
        assert document["section"] and document["section_path"]
        assert document["source_uri"].startswith("https://blob/corpus/raw/")
        assert document["content"].strip()
        assert len(document["content_vector"]) == embedder.dimensions


def test_page_number_is_populated_only_for_paginated_formats(pipeline, one_of_each, indexer) -> None:
    pipeline.run(one_of_each)

    by_type: dict[str, list[dict]] = {}
    for document in indexer.documents.values():
        by_type.setdefault(document["document_type"], []).append(document)

    assert all(d["page_number"] == 1 for d in by_type["pdf"])
    assert all(d["page_number"] is None for d in by_type["markdown"])


def test_ensure_index_runs_before_writing(pipeline, one_of_each, indexer) -> None:
    pipeline.run(one_of_each)

    assert indexer.ensure_index_calls == 1


# -- idempotency ---------------------------------------------------------------


def test_second_run_skips_unchanged_documents(pipeline, one_of_each, indexer, embedder) -> None:
    first = pipeline.run(one_of_each)
    keys_after_first = set(indexer.documents)
    tokens_after_first = embedder.total_tokens

    second = pipeline.run(one_of_each)

    assert first.docs_indexed == 4
    assert second.docs_indexed == 0
    assert second.docs_skipped == 4
    assert second.embedding_tokens == 0
    # Nothing was re-embedded and nothing in the index moved.
    assert embedder.total_tokens == tokens_after_first
    assert set(indexer.documents) == keys_after_first


def test_force_reindexes_without_duplicating_chunks(pipeline, one_of_each, indexer) -> None:
    pipeline.run(one_of_each)
    keys_after_first = set(indexer.documents)

    forced = pipeline.run(one_of_each, force=True)

    assert forced.docs_indexed == 4
    assert forced.docs_skipped == 0
    assert set(indexer.documents) == keys_after_first


def test_edited_document_is_reindexed(pipeline, tmp_path: Path, indexer) -> None:
    path = tmp_path / "cnt-hr-900_policy.md"
    path.write_text("# Policy\n\nCarryover is 5 days.\n", encoding="utf-8")
    pipeline.run([path])

    path.write_text("# Policy\n\nCarryover is 10 days.\n", encoding="utf-8")
    second = pipeline.run([path])

    assert second.docs_indexed == 1
    assert "10 days" in indexer.documents["CNT-HR-900-0000"]["content"]


def test_dry_run_touches_neither_embedder_nor_index(pipeline, one_of_each, indexer, embedder) -> None:
    report = pipeline.run(one_of_each, dry_run=True)

    assert report.docs_indexed == 4
    assert report.chunks_written == 4
    # Reported as work that *would* happen; nothing was actually written.
    assert indexer.documents == {}
    assert indexer.ensure_index_calls == 0
    assert embedder.total_tokens == 0


# -- re-index safety -----------------------------------------------------------


def test_shrinking_document_has_its_stale_chunks_deleted(
    embedder, indexer, tmp_path: Path
) -> None:
    """A document that loses content must not leave orphan chunks that keep
    answering queries."""
    settings = IngestionSettings(CORPUS_ROOT=tmp_path)
    pipeline = IngestionPipeline(settings=settings, embedder=embedder, indexer=indexer)
    path = tmp_path / "cnt-hr-900_policy.md"
    long_body = "\n\n".join(
        f"## Section {i}\n\n" + ("This clause governs the request process. " * 60)
        for i in range(4)
    )
    path.write_text(f"# Policy\n\n{long_body}\n", encoding="utf-8")

    first = pipeline.run([path])
    assert first.chunks_written > 1

    path.write_text("# Policy\n\nAll prior sections are withdrawn.\n", encoding="utf-8")
    second = pipeline.run([path])

    assert second.chunks_written == 1
    assert second.chunks_deleted == first.chunks_written - 1
    assert set(indexer.documents) == {"CNT-HR-900-0000"}
    assert "withdrawn" in indexer.documents["CNT-HR-900-0000"]["content"]


def test_growing_document_keeps_every_chunk(embedder, indexer, tmp_path: Path) -> None:
    settings = IngestionSettings(CORPUS_ROOT=tmp_path)
    pipeline = IngestionPipeline(settings=settings, embedder=embedder, indexer=indexer)
    path = tmp_path / "cnt-hr-900_policy.md"
    path.write_text("# Policy\n\nShort.\n", encoding="utf-8")
    pipeline.run([path])

    body = "\n\n".join(
        f"## Section {i}\n\n" + ("This clause governs the request process. " * 60)
        for i in range(3)
    )
    path.write_text(f"# Policy\n\n{body}\n", encoding="utf-8")
    second = pipeline.run([path])

    assert second.chunks_written > 1
    assert second.chunks_deleted == 0
    assert len(indexer.documents) == second.chunks_written


# -- error handling ------------------------------------------------------------


def test_one_bad_document_does_not_stop_the_run(pipeline, one_of_each, tmp_path: Path) -> None:
    broken = tmp_path / "cnt-hr-901_broken.docx"
    broken.write_bytes(b"definitely not a docx")

    report = pipeline.run([*one_of_each, broken])

    assert report.status == "failed"
    assert report.docs_indexed == 4
    assert report.docs_failed == 1
    failure = next(r for r in report.results if r.outcome is Outcome.FAILED)
    assert failure.source_path == str(broken)
    assert "cannot read docx" in failure.error


def test_unsupported_extension_is_reported_not_raised(pipeline, tmp_path: Path) -> None:
    path = tmp_path / "policy.rtf"
    path.write_text("x", encoding="utf-8")

    report = pipeline.run([path])

    assert report.docs_failed == 1
    assert "no loader" in report.results[0].error


def test_embedding_failure_leaves_the_previous_version_indexed(
    indexer, tmp_path: Path
) -> None:
    """A failed re-index must not blank out a working document."""

    class _FailingEmbedder(DeterministicEmbedder):
        fail = False

        def embed(self, texts):
            if self.fail:
                raise IngestionError("embedding backend unavailable")
            return super().embed(texts)

    embedder = _FailingEmbedder(dimensions=16)
    settings = IngestionSettings(CORPUS_ROOT=tmp_path)
    pipeline = IngestionPipeline(settings=settings, embedder=embedder, indexer=indexer)
    path = tmp_path / "cnt-hr-900_policy.md"
    path.write_text("# Policy\n\nOriginal text.\n", encoding="utf-8")
    pipeline.run([path])

    path.write_text("# Policy\n\nUpdated text.\n", encoding="utf-8")
    embedder.fail = True
    report = pipeline.run([path])

    assert report.docs_failed == 1
    assert "unavailable" in report.results[0].error
    assert "Original text." in indexer.documents["CNT-HR-900-0000"]["content"]


def test_run_aborts_when_prior_state_cannot_be_read(embedder, tmp_path: Path) -> None:
    """Silently defaulting to an empty hash map would re-embed the whole corpus."""

    class _UnreadableIndexer(InMemoryIndexer):
        def existing_hashes(self):
            raise IndexingError("search service unreachable")

    settings = IngestionSettings(CORPUS_ROOT=tmp_path)
    pipeline = IngestionPipeline(
        settings=settings, embedder=embedder, indexer=_UnreadableIndexer()
    )
    path = tmp_path / "cnt-hr-900_policy.md"
    path.write_text("# Policy\n\nBody.\n", encoding="utf-8")

    with pytest.raises(IndexingError, match="unreachable"):
        pipeline.run([path])


def test_empty_document_is_reported_as_failed(pipeline, tmp_path: Path) -> None:
    path = tmp_path / "cnt-hr-900_empty.md"
    path.write_text("   \n", encoding="utf-8")

    report = pipeline.run([path])

    assert report.docs_failed == 1


# -- reporting -----------------------------------------------------------------


def test_report_summary_matches_the_ingestion_run_record(pipeline, one_of_each) -> None:
    summary = pipeline.run(one_of_each).summary()

    assert summary["status"] == "succeeded"
    assert summary["docs_total"] == 4
    assert summary["docs_indexed"] == 4
    assert summary["embedding_tokens"] > 0
    assert summary["duration_ms"] >= 0
    assert set(summary) == {
        "status",
        "docs_total",
        "docs_indexed",
        "docs_skipped",
        "docs_failed",
        "chunks_written",
        "chunks_deleted",
        "embedding_tokens",
        "duration_ms",
    }


def test_chunking_settings_come_from_the_environment(monkeypatch) -> None:
    """The deployed job is configured only through environment variables."""
    monkeypatch.setenv("CHUNK_TARGET_TOKENS", "250")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "30")
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "300")

    settings = IngestionSettings()

    assert (settings.chunking.target_tokens, settings.chunking.overlap_tokens) == (250, 30)
    assert settings.chunking.max_tokens == 300


def test_azure_settings_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://svc.search.windows.net")
    monkeypatch.setenv("AZURE_SEARCH_INDEX_NAME", "custom-index")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("SOURCE_URI_BASE", "https://acct.blob.core.windows.net/corpus/raw")

    settings = IngestionSettings()

    assert settings.search.index_name == "custom-index"
    assert settings.search.endpoint == "https://svc.search.windows.net"
    assert settings.openai.embedding_dimensions == 1536
    assert settings.source_uri_base.endswith("/corpus/raw")


def test_invalid_chunking_configuration_is_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChunkingSettings(target_tokens=100, overlap_tokens=100)
    with pytest.raises(ValidationError):
        ChunkingSettings(target_tokens=800, overlap_tokens=100, max_tokens=400)


def test_chunk_size_configuration_changes_pipeline_output(embedder, tmp_path: Path) -> None:
    path = tmp_path / "cnt-hr-900_policy.md"
    body = "\n\n".join(
        f"## Section {i}\n\n" + ("This clause governs the request process. " * 40)
        for i in range(3)
    )
    path.write_text(f"# Policy\n\n{body}\n", encoding="utf-8")

    def build(target: int, overlap: int, maximum: int) -> IngestionPipeline:
        return IngestionPipeline(
            settings=IngestionSettings(
                CORPUS_ROOT=tmp_path,
                chunking=ChunkingSettings(
                    target_tokens=target, overlap_tokens=overlap, max_tokens=maximum
                ),
            ),
            embedder=embedder,
            indexer=InMemoryIndexer(),
        )

    small = build(200, 20, 250)
    large = build(4000, 100, 5000)

    assert small.chunker.target_tokens == 200
    assert small.run([path]).chunks_written > large.run([path]).chunks_written
    assert large.run([path], force=True).chunks_written == 1
