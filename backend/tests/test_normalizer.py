"""Normalization is where all four formats converge, so these tests assert the
converged shape rather than per-format behaviour."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.loaders import loader_for
from app.ingestion.models import Block, BlockKind, DocumentType, RawDocument
from app.ingestion.normalizer import normalize

ALL_TYPES = ["pdf", "docx", "html", "markdown"]

# Expected metadata per sample, from the corpus front matter.
EXPECTED = {
    "pdf": ("CNT-HR-005", "Paid Time Off Policy", DocumentType.PDF),
    "docx": ("CNT-FIN-016", "Expense Reimbursement Procedure", DocumentType.DOCX),
    "html": ("CNT-HR-023", "Employee Separation Procedure", DocumentType.HTML),
    "markdown": ("CNT-IT-024", "IT Equipment & Asset Policy", DocumentType.MARKDOWN),
}


def _normalize(path: Path, **kwargs):
    return normalize(loader_for(path).load(path), **kwargs)


@pytest.mark.parametrize("sample", ALL_TYPES, indirect=True)
def test_required_metadata_is_present_for_every_format(sample: Path, request) -> None:
    document_id, name, doc_type = EXPECTED[request.node.callspec.params["sample"]]
    meta = _normalize(sample).metadata

    assert meta.document_id == document_id
    assert meta.document_name == name
    assert meta.document_type is doc_type
    assert meta.source_uri
    assert meta.department and meta.category and meta.version and meta.status


@pytest.mark.parametrize("sample", ALL_TYPES, indirect=True)
def test_front_matter_is_lifted_out_of_the_body(sample: Path) -> None:
    """The `Document ID: ...` cover block is metadata, not retrievable content."""
    document = _normalize(sample)

    body = document.text
    assert "Document ID:" not in body
    assert "Confidentiality:" not in body
    assert "Approval Authority:" not in body


@pytest.mark.parametrize("sample", ALL_TYPES, indirect=True)
def test_every_block_carries_its_section_path(sample: Path) -> None:
    document = _normalize(sample)

    assert document.blocks
    for block in document.blocks:
        assert block.section_path, f"no section path on: {block.text[:40]}"
    # A heading's own path ends with itself.
    for block in document.blocks:
        if block.is_heading:
            assert block.section_path[-1] == block.text


def test_section_path_reflects_nesting() -> None:
    raw = RawDocument(
        blocks=[
            Block(kind=BlockKind.HEADING, text="Benefits", level=1),
            Block(kind=BlockKind.HEADING, text="PTO", level=2),
            Block(kind=BlockKind.HEADING, text="Carryover", level=3),
            Block(kind=BlockKind.PARAGRAPH, text="Up to 5 days carry over."),
            Block(kind=BlockKind.HEADING, text="Health", level=2),
            Block(kind=BlockKind.PARAGRAPH, text="Two plans are offered."),
        ],
        document_type=DocumentType.MARKDOWN,
        source_path="/tmp/benefits.md",
    )

    blocks = normalize(raw).blocks

    assert blocks[3].section_path == ("Benefits", "PTO", "Carryover")
    # Level 2 pops the level-3 sibling off the stack.
    assert blocks[5].section_path == ("Benefits", "Health")


@pytest.mark.parametrize("sample", ALL_TYPES, indirect=True)
def test_repeated_title_heading_is_collapsed(sample: Path) -> None:
    """Corpus documents print their title twice; only one heading should survive."""
    document = _normalize(sample)
    name = document.metadata.document_name
    titles = [b for b in document.blocks if b.is_heading and b.text == name]

    assert len(titles) == 1


def test_document_id_falls_back_to_filename(tmp_path: Path) -> None:
    path = tmp_path / "cnt-hr-099_mystery_policy.md"
    path.write_text("# Mystery Policy\n\nBody text here.\n", encoding="utf-8")

    assert _normalize(path).metadata.document_id == "CNT-HR-099"


def test_explicit_document_id_and_source_uri_win(tmp_path: Path) -> None:
    path = tmp_path / "cnt-hr-099_mystery.md"
    path.write_text("# Mystery\n\nBody.\n", encoding="utf-8")

    meta = _normalize(path, document_id="OVERRIDE-1", source_uri="https://blob/x.md").metadata

    assert meta.document_id == "OVERRIDE-1"
    assert meta.source_uri == "https://blob/x.md"


def test_source_uri_defaults_to_resolvable_file_uri(tmp_path: Path) -> None:
    path = tmp_path / "cnt-hr-099_a.md"
    path.write_text("# A\n\nBody.\n", encoding="utf-8")

    assert _normalize(path).metadata.source_uri.startswith("file://")


@pytest.mark.parametrize("sample", ALL_TYPES, indirect=True)
def test_content_hash_is_stable_across_runs(sample: Path) -> None:
    assert _normalize(sample).content_hash == _normalize(sample).content_hash


def test_content_hash_ignores_source_uri(tmp_path: Path) -> None:
    """Moving the same bytes to a new container must not force a re-embed."""
    path = tmp_path / "cnt-hr-099_a.md"
    path.write_text("# A\n\nBody.\n", encoding="utf-8")

    local = _normalize(path)
    moved = _normalize(path, source_uri="https://other/a.md")

    assert local.content_hash == moved.content_hash


def test_content_hash_changes_when_text_changes(tmp_path: Path) -> None:
    path = tmp_path / "cnt-hr-099_a.md"
    path.write_text("# A\n\nUp to 5 days carry over.\n", encoding="utf-8")
    before = _normalize(path).content_hash

    path.write_text("# A\n\nUp to 10 days carry over.\n", encoding="utf-8")

    assert _normalize(path).content_hash != before


def test_content_hash_changes_when_structure_changes(tmp_path: Path) -> None:
    """Identical prose under a different heading level is a different document."""
    path = tmp_path / "cnt-hr-099_a.md"
    path.write_text("# A\n\n## Scope\n\nApplies to staff.\n", encoding="utf-8")
    before = _normalize(path).content_hash

    path.write_text("# A\n\n### Scope\n\nApplies to staff.\n", encoding="utf-8")

    assert _normalize(path).content_hash != before


def test_normalization_collapses_whitespace_and_unicode(tmp_path: Path) -> None:
    path = tmp_path / "cnt-hr-099_a.md"
    # Non-breaking space and a soft hyphen, both common in exported policy text.
    path.write_text("# A\n\nEmployees may car­ry over days.\n", encoding="utf-8")

    body = next(b for b in _normalize(path).blocks if not b.is_heading)

    assert body.text == "Employees may carry over days."
