"""Chunking behaviour: section boundaries, configurable size, overlap, and the
metadata every chunk must carry."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.loaders import loader_for
from app.ingestion.models import (
    Block,
    BlockKind,
    DocumentType,
    NormalizedDocument,
    RawDocument,
)
from app.ingestion.normalizer import normalize
from app.ingestion.tokens import count_tokens

ALL_TYPES = ["pdf", "docx", "html", "markdown"]

SENTENCE = "Employees must submit the request before the leave begins whenever possible. "


def _document(blocks: list[Block], *, name: str = "Leave Policy") -> NormalizedDocument:
    return normalize(
        RawDocument(
            blocks=blocks,
            document_type=DocumentType.MARKDOWN,
            source_path=f"/tmp/cnt-hr-900_{name.lower().replace(' ', '_')}.md",
        )
    )


def _long_section(heading: str, sentences: int, level: int = 2) -> list[Block]:
    return [
        Block(kind=BlockKind.HEADING, text=heading, level=level),
        Block(kind=BlockKind.PARAGRAPH, text=(SENTENCE * sentences).strip()),
    ]


# -- metadata ------------------------------------------------------------------


@pytest.mark.parametrize("sample", ALL_TYPES, indirect=True)
def test_every_chunk_carries_required_metadata(sample: Path, chunker) -> None:
    document = normalize(loader_for(sample).load(sample))

    chunks = chunker.chunk(document)

    assert chunks
    for chunk in chunks:
        assert chunk.document_id == document.metadata.document_id
        assert chunk.document_name == document.metadata.document_name
        assert chunk.document_type is document.metadata.document_type
        assert chunk.source_uri == document.metadata.source_uri
        assert chunk.section and chunk.section_path
        assert chunk.token_count > 0
        assert chunk.content_hash == document.content_hash
        # page_number is populated only where the format exposes pagination.
        if document.metadata.document_type is DocumentType.PDF:
            assert chunk.page_number == 1
        else:
            assert chunk.page_number is None


@pytest.mark.parametrize("sample", ALL_TYPES, indirect=True)
def test_chunk_ids_are_deterministic_and_index_safe(sample: Path, chunker) -> None:
    document = normalize(loader_for(sample).load(sample))

    first = [c.chunk_id for c in chunker.chunk(document)]
    second = [c.chunk_id for c in chunker.chunk(document)]

    assert first == second
    assert len(set(first)) == len(first)
    for chunk_id in first:
        # Azure AI Search keys allow only these characters.
        assert all(c.isalnum() or c in "_-=" for c in chunk_id)


def test_embedding_text_carries_the_heading_breadcrumb(chunker) -> None:
    document = _document(
        [
            Block(kind=BlockKind.HEADING, text="Leave Policy", level=1),
            Block(kind=BlockKind.HEADING, text="Carryover", level=2),
            Block(kind=BlockKind.PARAGRAPH, text="Up to 5 unused days carry over."),
        ]
    )

    chunk = chunker.chunk(document)[0]

    assert chunk.embedding_text.startswith("Leave Policy > Carryover")
    # The breadcrumb is an embedding aid; the citable body stays verbatim.
    assert chunk.text == "Up to 5 unused days carry over."
    assert "Leave Policy" not in chunk.text


def test_breadcrumb_does_not_repeat_the_title(chunker) -> None:
    document = _document(
        [
            Block(kind=BlockKind.HEADING, text="Leave Policy", level=1),
            Block(kind=BlockKind.PARAGRAPH, text="Applies to all staff."),
        ]
    )

    assert chunker.chunk(document)[0].embedding_text.startswith("Leave Policy\n\n")


# -- structure awareness -------------------------------------------------------


def test_headings_survive_in_body_when_sections_are_packed(chunker) -> None:
    """Short sibling sections share a chunk, so their headings must be written
    into the text or the hierarchy is lost to the reader."""
    document = _document(
        [
            Block(kind=BlockKind.HEADING, text="Leave Policy", level=1),
            Block(kind=BlockKind.HEADING, text="Accrual", level=2),
            Block(kind=BlockKind.PARAGRAPH, text="Twenty days per year."),
            Block(kind=BlockKind.HEADING, text="Carryover", level=2),
            Block(kind=BlockKind.PARAGRAPH, text="Five days carry over."),
        ]
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert "Accrual" in chunks[0].text and "Carryover" in chunks[0].text
    # Cited to the shared ancestor, which is a real location in the document.
    assert chunks[0].section_path == ("Leave Policy",)


def test_chunks_never_cross_a_top_level_heading(chunker) -> None:
    document = _document(
        [
            Block(kind=BlockKind.HEADING, text="Leave", level=1),
            Block(kind=BlockKind.PARAGRAPH, text="Twenty days per year."),
            Block(kind=BlockKind.HEADING, text="Security", level=1),
            Block(kind=BlockKind.PARAGRAPH, text="Use approved devices only."),
        ]
    )

    chunks = chunker.chunk(document)

    assert [c.section_path for c in chunks] == [("Leave",), ("Security",)]
    assert "Security" not in chunks[0].text


def test_large_sections_become_separate_chunks(chunker) -> None:
    """When sections exceed the target the chunker degrades to section-level
    granularity on its own, with no second mode."""
    document = _document(
        [
            Block(kind=BlockKind.HEADING, text="Leave Policy", level=1),
            *_long_section("Accrual", 45),
            *_long_section("Carryover", 45),
        ]
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 2
    assert chunks[0].section == "Accrual"
    assert chunks[1].section == "Carryover"


def test_table_is_never_split(corpus: Path, chunker) -> None:
    path = corpus / "markdown" / "cnt-hr-021_manager_approval_matrix.md"
    document = normalize(loader_for(path).load(path))

    chunks = chunker.chunk(document)
    with_table = [c for c in chunks if "| Request |" in c.text]

    assert len(with_table) == 1
    assert "Accommodation request" in with_table[0].text


def test_oversized_table_is_emitted_whole_rather_than_cut() -> None:
    rows = "\n".join(f"| Item {i} | Approver {i} |" for i in range(400))
    document = _document(
        [
            Block(kind=BlockKind.HEADING, text="Matrix", level=1),
            Block(kind=BlockKind.TABLE, text=f"| A | B |\n|---|---|\n{rows}"),
        ]
    )

    chunks = StructureAwareChunker(target_tokens=200, overlap_tokens=20, max_tokens=250).chunk(
        document
    )

    assert len(chunks) == 1
    assert "Item 0" in chunks[0].text and "Item 399" in chunks[0].text


def test_heading_only_section_is_not_emitted_as_an_empty_chunk(chunker) -> None:
    document = _document(
        [
            Block(kind=BlockKind.HEADING, text="Leave Policy", level=1),
            Block(kind=BlockKind.HEADING, text="Reserved", level=2),
            Block(kind=BlockKind.HEADING, text="Accrual", level=2),
            Block(kind=BlockKind.PARAGRAPH, text="Twenty days per year."),
        ]
    )

    chunks = chunker.chunk(document)

    assert all(c.text.strip() for c in chunks)
    # The empty heading is gone, but the populated one is still reachable.
    assert any("Accrual" in c.section_path or c.section == "Accrual" for c in chunks)


# -- configurable size and overlap ---------------------------------------------


def test_chunk_size_is_configurable(chunker) -> None:
    document = _document(
        [
            Block(kind=BlockKind.HEADING, text="Leave Policy", level=1),
            *_long_section("A", 12),
            *_long_section("B", 12),
            *_long_section("C", 12),
        ]
    )

    small = StructureAwareChunker(target_tokens=200, overlap_tokens=20, max_tokens=300)
    large = StructureAwareChunker(target_tokens=2000, overlap_tokens=100, max_tokens=3000)

    assert len(small.chunk(document)) > len(large.chunk(document))
    assert len(large.chunk(document)) == 1


def test_no_chunk_exceeds_max_tokens_when_splittable() -> None:
    document = _document(
        [
            Block(kind=BlockKind.HEADING, text="Leave Policy", level=1),
            *_long_section("Accrual", 200),
        ]
    )

    chunker = StructureAwareChunker(target_tokens=300, overlap_tokens=50, max_tokens=400)
    chunks = chunker.chunk(document)

    assert len(chunks) > 1
    assert all(c.token_count <= 400 for c in chunks)


def test_overlap_repeats_context_between_splits_of_one_section() -> None:
    document = _document(
        [
            Block(kind=BlockKind.HEADING, text="Leave Policy", level=1),
            *_long_section("Accrual", 120),
        ]
    )

    with_overlap = StructureAwareChunker(target_tokens=300, overlap_tokens=100, max_tokens=350)
    without = StructureAwareChunker(target_tokens=300, overlap_tokens=0, max_tokens=350)

    overlapped = with_overlap.chunk(document)
    plain = without.chunk(document)

    assert len(overlapped) > 1
    # Overlap duplicates the tail of each part into the next, so total tokens
    # rise while the underlying section is unchanged.
    assert sum(c.token_count for c in overlapped) > sum(c.token_count for c in plain)
    # The tail of chunk 0 reappears at the head of chunk 1.
    tail = overlapped[0].text.split(". ")[-2]
    assert tail in overlapped[1].text


def test_overlap_must_be_smaller_than_target() -> None:
    with pytest.raises(ValueError, match="overlap_tokens must be smaller"):
        StructureAwareChunker(target_tokens=100, overlap_tokens=100)


def test_max_tokens_must_not_be_below_target() -> None:
    with pytest.raises(ValueError, match="max_tokens must be >="):
        StructureAwareChunker(target_tokens=800, overlap_tokens=100, max_tokens=400)


def test_token_count_matches_the_embedded_text(chunker) -> None:
    document = _document(
        [
            Block(kind=BlockKind.HEADING, text="Leave Policy", level=1),
            Block(kind=BlockKind.PARAGRAPH, text="Twenty days per year."),
        ]
    )

    chunk = chunker.chunk(document)[0]

    assert chunk.token_count == count_tokens(chunk.embedding_text)


def test_ordinals_are_contiguous_from_zero(chunker) -> None:
    document = _document(
        [
            Block(kind=BlockKind.HEADING, text="Leave Policy", level=1),
            *_long_section("A", 45),
            *_long_section("B", 45),
            *_long_section("C", 45),
        ]
    )

    assert [c.ordinal for c in chunker.chunk(document)] == [0, 1, 2]
