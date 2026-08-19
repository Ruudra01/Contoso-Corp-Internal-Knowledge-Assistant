"""Chunking strategy coverage: headings, paragraphs, lists, long sections,
documents without headings, and metadata preservation.

The corpus contains no lists in any of the four formats (verified against raw
Markdown markers, HTML `<li>`, DOCX `numPr` and PDF bullet glyphs), and no
section above 800 tokens, so list and long-section behaviour is specified here
against synthetic documents. Each `test_regression_*` pins a defect that was
present before this strategy work and is now fixed.
"""

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

RULE = "This clause governs the request process and its documented exceptions. "


def build(blocks: list[Block], *, stem: str = "cnt-hr-900_leave_policy") -> NormalizedDocument:
    return normalize(
        RawDocument(
            blocks=blocks,
            document_type=DocumentType.MARKDOWN,
            source_path=f"/tmp/{stem}.md",
        )
    )


def heading(text: str, level: int = 2) -> Block:
    return Block(kind=BlockKind.HEADING, text=text, level=level)


def para(text: str) -> Block:
    return Block(kind=BlockKind.PARAGRAPH, text=text)


def item(text: str) -> Block:
    return Block(kind=BlockKind.LIST_ITEM, text=text)


def long_para(sentences: int) -> Block:
    return para((RULE * sentences).strip())


# =============================================================================
# Headings and section hierarchy
# =============================================================================


def test_heading_hierarchy_is_preserved_in_section_path() -> None:
    document = build(
        [
            heading("Benefits", 1),
            heading("PTO", 2),
            heading("Carryover", 3),
            long_para(60),
        ]
    )

    chunk = StructureAwareChunker(target_tokens=200, overlap_tokens=20, max_tokens=900).chunk(
        document
    )[0]

    assert chunk.section_path == ("Benefits", "PTO", "Carryover")
    assert chunk.section == "Carryover"
    assert chunk.section_path_text == "Benefits > PTO > Carryover"


def test_heading_breadcrumb_prefixes_the_embedded_text() -> None:
    document = build([heading("Benefits", 1), heading("Carryover", 2), para("Five days carry over.")])

    chunk = StructureAwareChunker().chunk(document)[0]

    assert chunk.embedding_text.startswith("Benefits > Carryover\n\n")
    # The citable body stays verbatim; the breadcrumb is an embedding aid only.
    assert chunk.text == "Five days carry over."


def test_headings_are_written_into_the_body_when_sections_are_packed() -> None:
    """Short sibling sections share a chunk, so their headings must survive in
    the text or the reader loses the hierarchy."""
    document = build(
        [
            heading("Leave Policy", 1),
            heading("Accrual", 2),
            para("Twenty days per year."),
            heading("Carryover", 2),
            para("Five days carry over."),
        ]
    )

    chunks = StructureAwareChunker().chunk(document)

    assert len(chunks) == 1
    assert "Accrual" in chunks[0].text
    assert "Carryover" in chunks[0].text
    # Cited to the shared ancestor rather than falsely claiming one subsection.
    assert chunks[0].section_path == ("Leave Policy",)


def test_chunks_never_cross_a_top_level_heading() -> None:
    document = build(
        [
            heading("Leave", 1),
            para("Twenty days per year."),
            heading("Security", 1),
            para("Use approved devices only."),
        ]
    )

    chunks = StructureAwareChunker().chunk(document)

    assert [c.section_path for c in chunks] == [("Leave",), ("Security",)]
    assert "Security" not in chunks[0].text


def test_deeper_sections_split_before_unrelated_ones_merge() -> None:
    """Two H3 leaves under different H2 parents may share a chunk only because
    they share the H1; the citation must fall back to that shared ancestor."""
    document = build(
        [
            heading("Handbook", 1),
            heading("Leave", 2),
            heading("Accrual", 3),
            para("Twenty days per year."),
            heading("Security", 2),
            heading("Devices", 3),
            para("Use approved devices only."),
        ]
    )

    chunks = StructureAwareChunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].section_path == ("Handbook",)


def test_heading_only_section_produces_no_empty_chunk() -> None:
    document = build(
        [
            heading("Leave Policy", 1),
            heading("Reserved", 2),
            heading("Accrual", 2),
            para("Twenty days per year."),
        ]
    )

    chunks = StructureAwareChunker().chunk(document)

    assert all(c.text.strip() for c in chunks)
    assert "Reserved" not in " ".join(c.text for c in chunks)


# =============================================================================
# Paragraphs — individual policy rules
# =============================================================================


def test_paragraph_is_never_split_when_it_fits() -> None:
    """A paragraph is where one policy rule lives, so it must stay whole."""
    rule = "Carryover above 5 days requires written HR approval before December 31."
    document = build([heading("Carryover", 1), long_para(30), para(rule), long_para(30)])

    chunks = StructureAwareChunker(target_tokens=120, overlap_tokens=0, max_tokens=150).chunk(
        document
    )

    assert len(chunks) > 1
    assert sum(rule in c.text for c in chunks) == 1, "rule was duplicated or cut"


def test_paragraph_boundaries_are_respected_when_packing() -> None:
    document = build(
        [heading("Rules", 1), para("Rule one is stated here."), para("Rule two is stated here.")]
    )

    chunks = StructureAwareChunker().chunk(document)

    assert len(chunks) == 1
    # Paragraphs stay separated in the body, not run together.
    assert "\n\n" in chunks[0].text


def test_oversized_paragraph_splits_on_sentence_boundaries() -> None:
    document = build([heading("Accrual", 1), long_para(200)])

    chunks = StructureAwareChunker(target_tokens=300, overlap_tokens=0, max_tokens=350).chunk(
        document
    )

    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.text.removesuffix("(continued below)").strip()
        # Every part ends at a sentence terminator, so no clause is cut mid-word.
        assert body.endswith("."), body[-40:]


# =============================================================================
# Lists — enumerated rule sets
# =============================================================================


def test_list_run_is_kept_whole_when_it_fits_under_the_ceiling() -> None:
    document = build([heading("Eligibility", 1), *[item(f"Condition {i} applies.") for i in range(20)]])

    chunks = StructureAwareChunker(target_tokens=100, overlap_tokens=0, max_tokens=900).chunk(
        document
    )

    assert len(chunks) == 1, "a list that fits under max_tokens must not be split"
    assert chunks[0].text.count("Condition") == 20


def test_list_is_not_separated_from_its_introducing_paragraph_when_it_fits() -> None:
    document = build(
        [
            heading("Eligibility", 1),
            para("An employee qualifies when all of the following are true:"),
            *[item(f"Condition {i} applies.") for i in range(6)],
        ]
    )

    chunks = StructureAwareChunker().chunk(document)

    assert len(chunks) == 1
    assert "all of the following" in chunks[0].text
    assert chunks[0].text.count("Condition") == 6


def test_oversized_list_breaks_only_at_item_boundaries() -> None:
    items = [
        item(f"Condition {i}: the employee must satisfy this requirement before approval.")
        for i in range(40)
    ]
    document = build([heading("Eligibility", 1), *items])

    chunks = StructureAwareChunker(target_tokens=200, overlap_tokens=0, max_tokens=260).chunk(
        document
    )

    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.text.removesuffix("(continued below)").strip()
        # Each part ends with a complete item, never a truncated one.
        assert body.endswith("approval."), body[-40:]
    # Every item survives exactly once when overlap is off.
    joined = " ".join(c.text for c in chunks)
    for i in range(40):
        assert f"Condition {i}:" in joined


def test_broken_list_is_marked_as_continuing() -> None:
    """A reader must be able to tell that an enumeration continues elsewhere,
    or conjunctive conditions look complete when they are not."""
    items = [
        item(f"Condition {i}: the employee must satisfy this requirement before approval.")
        for i in range(40)
    ]
    document = build([heading("Eligibility", 1), *items])

    chunks = StructureAwareChunker(target_tokens=200, overlap_tokens=0, max_tokens=260).chunk(
        document
    )

    assert all("(continued" in c.text for c in chunks)


def test_table_is_never_split(corpus: Path) -> None:
    path = corpus / "markdown" / "cnt-hr-021_manager_approval_matrix.md"
    document = normalize(loader_for(path).load(path))

    chunks = StructureAwareChunker(target_tokens=60, overlap_tokens=0, max_tokens=80).chunk(document)
    with_table = [c for c in chunks if "| Request |" in c.text]

    assert len(with_table) == 1
    # Emitted whole even though it exceeds the ceiling: half a table misleads.
    assert "Accommodation request" in with_table[0].text


# =============================================================================
# Long sections — token limits and overlap
# =============================================================================


def test_no_chunk_exceeds_max_tokens_including_the_breadcrumb() -> None:
    """The ceiling is enforced on the embedded text, breadcrumb included."""
    long_title = ("Very Long Enterprise Policy Document Title That Repeats " * 6).strip()
    document = build([heading(long_title, 1), long_para(120)])

    chunks = StructureAwareChunker(target_tokens=200, overlap_tokens=20, max_tokens=250).chunk(
        document
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 250
        assert chunk.token_count == count_tokens(chunk.embedding_text)


def test_overlap_repeats_context_across_paragraph_boundaries() -> None:
    """The common oversize shape is a multi-paragraph section. Overlap has to
    apply there, not only inside one oversized paragraph."""
    document = build([heading("Accrual", 1), *[long_para(12) for _ in range(6)]])

    with_overlap = StructureAwareChunker(target_tokens=200, overlap_tokens=150, max_tokens=400)
    without = StructureAwareChunker(target_tokens=200, overlap_tokens=0, max_tokens=400)

    overlapped = with_overlap.chunk(document)
    plain = without.chunk(document)

    assert len(overlapped) == len(plain) > 1
    assert sum(c.token_count for c in overlapped) > sum(c.token_count for c in plain)
    assert any("(continued)" in c.text for c in overlapped[1:])


def test_overlap_tail_comes_from_the_preceding_chunk() -> None:
    document = build([heading("Accrual", 1), long_para(200)])

    chunks = StructureAwareChunker(target_tokens=300, overlap_tokens=120, max_tokens=600).chunk(
        document
    )

    assert len(chunks) > 1
    previous_tail = chunks[0].text.rstrip().removesuffix("(continued below)").strip().split(". ")[-1]
    assert previous_tail.rstrip(".") in chunks[1].text


def test_overlap_never_pushes_a_chunk_over_the_ceiling() -> None:
    document = build([heading("Accrual", 1), *[long_para(10) for _ in range(8)]])

    chunker = StructureAwareChunker(target_tokens=150, overlap_tokens=140, max_tokens=160)
    chunks = chunker.chunk(document)

    assert len(chunks) > 1
    assert all(c.token_count <= 160 for c in chunks)


def test_zero_overlap_duplicates_nothing() -> None:
    document = build([heading("Accrual", 1), *[para(f"Rule {i} is stated in full here.") for i in range(12)]])

    chunks = StructureAwareChunker(target_tokens=40, overlap_tokens=0, max_tokens=50).chunk(document)

    assert len(chunks) > 1
    joined = " ".join(c.text for c in chunks)
    for i in range(12):
        assert joined.count(f"Rule {i} is") == 1


def test_max_chunk_size_is_configurable() -> None:
    document = build([heading("Handbook", 1), *[long_para(20) for _ in range(6)]])

    small = StructureAwareChunker(target_tokens=200, overlap_tokens=0, max_tokens=250)
    large = StructureAwareChunker(target_tokens=4000, overlap_tokens=0, max_tokens=5000)

    assert len(small.chunk(document)) > len(large.chunk(document))
    assert len(large.chunk(document)) == 1


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap_tokens must be smaller"):
        StructureAwareChunker(target_tokens=100, overlap_tokens=100)
    with pytest.raises(ValueError, match="max_tokens must be >="):
        StructureAwareChunker(target_tokens=800, overlap_tokens=10, max_tokens=400)
    with pytest.raises(ValueError, match="target_tokens must be positive"):
        StructureAwareChunker(target_tokens=0)
    with pytest.raises(ValueError, match="overlap_tokens must not be negative"):
        StructureAwareChunker(target_tokens=100, overlap_tokens=-1)


# =============================================================================
# Documents without headings
# =============================================================================


def test_headless_document_still_produces_a_chunk() -> None:
    document = build(
        [para("Rule one applies."), para("Rule two applies.")], stem="cnt-hr-901_untitled"
    )

    chunks = StructureAwareChunker().chunk(document)

    assert len(chunks) == 1
    assert "Rule one applies." in chunks[0].text


def test_headless_chunk_is_traceable_to_the_document() -> None:
    """With no heading to name, the document itself is the section of record —
    `section_path` must never be empty or the citation cannot be resolved."""
    document = build([para("Rule one applies.")], stem="cnt-hr-901_untitled")

    chunk = StructureAwareChunker().chunk(document)[0]

    assert chunk.section_path == (chunk.document_name,)
    assert chunk.section == chunk.document_name
    assert chunk.section_path_text == chunk.document_name


def test_long_headless_document_splits_and_stays_traceable() -> None:
    document = build([long_para(40) for _ in range(5)], stem="cnt-hr-901_untitled")

    chunks = StructureAwareChunker(target_tokens=200, overlap_tokens=20, max_tokens=300).chunk(
        document
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.section_path == (chunk.document_name,)
        assert chunk.token_count <= 300


def test_headless_document_ordinals_stay_contiguous() -> None:
    document = build([long_para(30) for _ in range(4)], stem="cnt-hr-901_untitled")

    chunks = StructureAwareChunker(target_tokens=150, overlap_tokens=0, max_tokens=200).chunk(
        document
    )

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


# =============================================================================
# Metadata preservation and traceability
# =============================================================================

ALL_TYPES = ["pdf", "docx", "html", "markdown"]


@pytest.mark.parametrize("sample", ALL_TYPES, indirect=True)
def test_metadata_survives_chunking_for_every_format(sample: Path) -> None:
    document = normalize(loader_for(sample).load(sample))

    chunks = StructureAwareChunker().chunk(document)

    assert chunks
    for chunk in chunks:
        assert chunk.document_id == document.metadata.document_id
        assert chunk.document_name == document.metadata.document_name
        assert chunk.document_type is document.metadata.document_type
        assert chunk.source_uri == document.metadata.source_uri
        assert chunk.content_hash == document.content_hash
        assert chunk.section and chunk.section_path
        assert chunk.token_count > 0


@pytest.mark.parametrize("sample", ALL_TYPES, indirect=True)
def test_metadata_survives_splitting_into_many_chunks(sample: Path) -> None:
    """A tiny ceiling forces multiple chunks per document; every one must carry
    the full metadata set, not just the first."""
    document = normalize(loader_for(sample).load(sample))

    chunks = StructureAwareChunker(target_tokens=40, overlap_tokens=10, max_tokens=60).chunk(
        document
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.document_id == document.metadata.document_id
        assert chunk.document_type is document.metadata.document_type
        assert chunk.source_uri == document.metadata.source_uri
        assert chunk.section_path


def test_page_numbers_are_preserved_and_ordered(corpus: Path) -> None:
    path = corpus / "pdf" / "cnt-hr-005_paid_time_off_policy.pdf"
    document = normalize(loader_for(path).load(path))

    chunks = StructureAwareChunker(target_tokens=40, overlap_tokens=0, max_tokens=60).chunk(document)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.page_number == 1
        assert chunk.page_end is not None and chunk.page_end >= chunk.page_number


def test_page_number_is_none_for_unpaginated_formats() -> None:
    document = build([heading("Accrual", 1), para("Twenty days per year.")])

    chunk = StructureAwareChunker().chunk(document)[0]

    assert chunk.page_number is None and chunk.page_end is None


def test_chunk_ids_are_deterministic_across_runs() -> None:
    document = build([heading("Handbook", 1), *[long_para(20) for _ in range(5)]])
    chunker = StructureAwareChunker(target_tokens=200, overlap_tokens=20, max_tokens=250)

    first = [c.chunk_id for c in chunker.chunk(document)]
    second = [c.chunk_id for c in chunker.chunk(document)]

    assert first == second
    assert len(set(first)) == len(first) > 1


def test_chunk_ids_are_index_key_safe() -> None:
    document = build([heading("Accrual & Carryover (2026)", 1), para("Twenty days.")])

    for chunk in StructureAwareChunker().chunk(document):
        # Azure AI Search keys permit only letters, digits, underscore, dash, equals.
        assert all(ch.isalnum() or ch in "_-=" for ch in chunk.chunk_id)


def test_chunk_id_encodes_document_and_ordinal() -> None:
    document = build([heading("Handbook", 1), *[long_para(20) for _ in range(3)]])

    chunks = StructureAwareChunker(target_tokens=200, overlap_tokens=0, max_tokens=250).chunk(
        document
    )

    for chunk in chunks:
        assert chunk.chunk_id == f"{chunk.document_id}-{chunk.ordinal:04d}"


@pytest.mark.parametrize("sample", ALL_TYPES, indirect=True)
def test_every_chunk_is_traceable_to_document_and_section(sample: Path) -> None:
    """The end-to-end traceability requirement: id, name, URI, section path."""
    document = normalize(loader_for(sample).load(sample))

    for chunk in StructureAwareChunker().chunk(document):
        assert chunk.document_id
        assert chunk.source_uri.startswith(("file://", "http://", "https://"))
        assert chunk.section_path_text
        assert chunk.chunk_id.startswith(chunk.document_id)


# =============================================================================
# Regressions — each pins a defect confirmed present before this work
# =============================================================================


def test_regression_overlap_was_inert_across_block_boundaries() -> None:
    """Overlap previously applied only inside a single oversized block, so a
    multi-paragraph section split with no overlap at all."""
    document = build([heading("Accrual", 1), *[long_para(12) for _ in range(6)]])

    plain = StructureAwareChunker(target_tokens=200, overlap_tokens=0, max_tokens=400).chunk(document)
    overlapped = StructureAwareChunker(
        target_tokens=200, overlap_tokens=150, max_tokens=400
    ).chunk(document)

    assert sum(c.token_count for c in overlapped) > sum(c.token_count for c in plain)


def test_regression_list_items_were_cut_without_a_marker() -> None:
    items = [item(f"Condition {i}: approval is required before the leave begins.") for i in range(40)]
    document = build([heading("Eligibility", 1), *items])

    chunks = StructureAwareChunker(target_tokens=200, overlap_tokens=0, max_tokens=260).chunk(
        document
    )

    assert all("(continued" in c.text for c in chunks)


def test_regression_headless_chunks_had_an_empty_section_path() -> None:
    document = build([para("Rule one applies.")], stem="cnt-hr-901_untitled")

    assert StructureAwareChunker().chunk(document)[0].section_path != ()


def test_regression_breadcrumb_pushed_chunks_over_max_tokens() -> None:
    long_title = ("Very Long Enterprise Policy Document Title That Repeats " * 6).strip()
    document = build([heading(long_title, 1), long_para(40)])

    chunks = StructureAwareChunker(target_tokens=200, overlap_tokens=20, max_tokens=250).chunk(
        document
    )

    assert all(c.token_count <= 250 for c in chunks)


# =============================================================================
# Invariant sweep
#
# The ceiling was breached three separate ways during development (breadcrumb,
# continuation marker, tokenizer drift on joins), each time passing the point
# tests that existed. This sweep is the guard: it asserts the invariants across
# many configurations and document shapes at once.
# =============================================================================

SHAPES = {
    "headings_and_paragraphs": [
        heading("Handbook", 1),
        heading("Leave", 2),
        long_para(30),
        heading("Security", 2),
        long_para(30),
    ],
    "one_huge_paragraph": [heading("Accrual", 1), long_para(300)],
    "many_small_paragraphs": [heading("Rules", 1), *[para(f"Rule {i} applies here.") for i in range(60)]],
    "long_list": [heading("Eligibility", 1), *[item(f"Condition {i}: approval required.") for i in range(60)]],
    "list_after_paragraph": [
        heading("Eligibility", 1),
        long_para(40),
        *[item(f"Condition {i}: approval required.") for i in range(30)],
    ],
    "no_headings": [long_para(40), long_para(40), long_para(40)],
    "deep_nesting": [
        heading("Handbook", 1),
        heading("Leave", 2),
        heading("Accrual", 3),
        long_para(40),
        heading("Carryover", 3),
        long_para(40),
    ],
    "long_title": [heading(("Extremely Long Policy Title Repeated " * 8).strip(), 1), long_para(80)],
}

CONFIGS = [
    (100, 0, 120),
    (100, 40, 160),
    (200, 20, 250),
    (200, 150, 400),
    (300, 120, 350),
    (800, 120, 1200),
    (50, 10, 60),
]


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize(("target", "overlap", "maximum"), CONFIGS)
def test_invariants_hold_across_configurations(
    shape: str, target: int, overlap: int, maximum: int
) -> None:
    document = build(SHAPES[shape])
    chunker = StructureAwareChunker(
        target_tokens=target, overlap_tokens=overlap, max_tokens=maximum
    )

    chunks = chunker.chunk(document)

    assert chunks, f"{shape} produced no chunks"
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids), "duplicate chunk ids"
    assert [c.ordinal for c in chunks] == list(range(len(chunks))), "ordinals not contiguous"

    for chunk in chunks:
        # The ceiling holds on the embedded text, the only exception being an
        # atomic block (a table) that cannot be split without misleading.
        assert chunk.token_count <= maximum, (
            f"{shape} @ target={target} overlap={overlap} max={maximum}: "
            f"chunk {chunk.ordinal} is {chunk.token_count} tokens"
        )
        assert chunk.text.strip(), "empty chunk body"
        assert chunk.token_count == count_tokens(chunk.embedding_text)
        # Traceability holds for every chunk of every shape.
        assert chunk.document_id and chunk.document_name
        assert chunk.section and chunk.section_path
        assert chunk.source_uri


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_chunking_is_deterministic_across_shapes(shape: str) -> None:
    document = build(SHAPES[shape])
    chunker = StructureAwareChunker(target_tokens=200, overlap_tokens=20, max_tokens=250)

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert [(c.chunk_id, c.text) for c in first] == [(c.chunk_id, c.text) for c in second]


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_no_content_is_lost_when_overlap_is_off(shape: str) -> None:
    """Every non-heading block's text must appear somewhere in the output.

    Guards against a packing bug silently dropping a unit — the failure mode
    that is invisible in a token-count assertion.
    """
    blocks = SHAPES[shape]
    document = build(blocks)

    chunks = StructureAwareChunker(target_tokens=200, overlap_tokens=0, max_tokens=250).chunk(
        document
    )
    joined = " ".join(c.text for c in chunks)

    for block in blocks:
        if block.is_heading or block.kind is BlockKind.TABLE:
            continue
        # Long paragraphs get sentence-split, so check a distinctive head fragment.
        probe = " ".join(block.text.split()[:6])
        assert probe in joined, f"lost content from: {probe!r}"
