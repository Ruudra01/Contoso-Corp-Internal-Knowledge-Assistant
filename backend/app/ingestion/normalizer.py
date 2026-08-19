"""Normalization: format-specific quirks in, one canonical document out.

Runs after any loader and knows nothing about PDF/DOCX/HTML/Markdown. It

* NFKC-normalizes text and strips the artefacts parsers leave behind;
* lifts the corpus front-matter block (`Document ID: CNT-HR-005`, ...) out of the
  body into typed metadata, so it stops polluting retrieval;
* drops the repeated title these documents carry;
* assigns every block its heading breadcrumb (`section_path`);
* computes the SHA-256 content hash that makes re-ingestion idempotent.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from app.core.logging import get_logger
from app.ingestion.models import (
    Block,
    DocumentMetadata,
    DocumentType,
    NormalizedDocument,
    RawDocument,
    content_hash,
)

logger = get_logger(__name__)

# Front-matter labels used across the corpus. Order matters only for readability;
# the regex alternation is longest-first to stop `Effective` eating
# `Effective Date`.
_FRONT_MATTER_LABELS = (
    "Document Title",
    "Document ID",
    "Department",
    "Policy Category",
    "Effective Date",
    "Last Updated",
    "Approval Authority",
    "Confidentiality",
    "Effective",
    "Version",
    "Owner",
    "Status",
    "Format",
    "Title",
)
_LABEL_ALT = "|".join(sorted(_FRONT_MATTER_LABELS, key=len, reverse=True))
_FRONT_MATTER = re.compile(
    rf"(?P<key>{_LABEL_ALT})\s*:\s*(?P<value>.*?)(?=\s*(?:{_LABEL_ALT})\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
# `cnt-hr-005_paid_time_off_policy.pdf` -> `CNT-HR-005`
_FILENAME_ID = re.compile(r"^(?P<id>[a-z]{2,4}-[a-z]{2,4}-\d{2,4})", re.IGNORECASE)
# Only scan the head of a document for front matter; a `Status:` deep in the body
# is real content.
_FRONT_MATTER_WINDOW = 20
_SOFT_HYPHEN = "­"


def normalize(
    raw: RawDocument,
    *,
    source_uri: str | None = None,
    document_id: str | None = None,
) -> NormalizedDocument:
    """Turn loader output into a `NormalizedDocument`."""
    blocks = [b for b in (_clean_block(b) for b in raw.blocks) if b is not None]

    front_matter, blocks = _extract_front_matter(blocks)
    blocks = _drop_repeated_title(blocks)
    _assign_section_paths(blocks)

    path = Path(raw.source_path)
    resolved_id = (
        document_id
        or _clean_value(front_matter.get("document id"))
        or _id_from_filename(path)
        or path.stem
    ).upper()
    name = (
        _clean_value(front_matter.get("document title"))
        or _clean_value(front_matter.get("title"))
        or raw.title_hint
        # The document's own top heading beats the filename, and is available even
        # when a loader did not volunteer a title hint.
        or next((b.text for b in blocks if b.is_heading and b.level == 1), None)
        or path.stem.replace("_", " ").title()
    )

    metadata = DocumentMetadata(
        document_id=resolved_id,
        document_name=name,
        document_type=raw.document_type,
        source_uri=source_uri or path.resolve().as_uri(),
        department=_clean_value(front_matter.get("department")),
        category=_clean_value(front_matter.get("policy category")),
        version=_clean_value(front_matter.get("version")),
        effective_date=_clean_value(
            front_matter.get("effective date") or front_matter.get("effective")
        ),
        status=_clean_value(front_matter.get("status")),
    )

    if not blocks:
        logger.warning(
            "document normalized to zero content blocks",
            extra={"document_id": resolved_id, "source": raw.source_path},
        )

    document = NormalizedDocument(
        metadata=metadata,
        blocks=blocks,
        content_hash=_document_hash(resolved_id, blocks),
        page_count=raw.page_count,
    )
    logger.debug(
        "normalized document",
        extra={
            "document_id": resolved_id,
            "document_type": str(raw.document_type),
            "blocks": len(blocks),
            "content_hash": document.content_hash[:12],
        },
    )
    return document


def _clean_block(block: Block) -> Block | None:
    text = _clean_text(block.text)
    if not text:
        return None
    block.text = text
    if block.is_heading and block.level is None:
        block.level = 2
    return block


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace(_SOFT_HYPHEN, "")
    # Collapse runs of whitespace but keep newlines, which tables depend on.
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _clean_value(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = " ".join(value.replace("*", "").split()).strip(" .;,")
    return cleaned or None


def _extract_front_matter(blocks: list[Block]) -> tuple[dict[str, str], list[Block]]:
    """Pull `Key: Value` metadata out of the document head.

    A block is removed only when it is *entirely* front matter — the loaders
    collapse the whole block differently per format (Markdown yields one
    paragraph, PDF yields one per line), so both shapes are handled by measuring
    how much of the block the matches cover.
    """
    found: dict[str, str] = {}
    kept: list[Block] = []

    for index, block in enumerate(blocks):
        if index >= _FRONT_MATTER_WINDOW or block.is_heading or block.is_atomic:
            kept.append(block)
            continue

        matches = list(_FRONT_MATTER.finditer(block.text))
        covered = sum(m.end() - m.start() for m in matches)
        if matches and covered >= 0.9 * len(block.text.strip()):
            for match in matches:
                key = " ".join(match.group("key").split()).lower()
                found.setdefault(key, match.group("value"))
            continue
        kept.append(block)

    return found, kept


def _drop_repeated_title(blocks: list[Block]) -> list[Block]:
    """These documents print their title twice (cover line, then body heading).
    Keep the later one so it owns the body that follows it."""
    level_one = [i for i, b in enumerate(blocks) if b.is_heading and b.level == 1]
    if len(level_one) < 2:
        return blocks
    first, second = level_one[0], level_one[1]
    if blocks[first].text.casefold() != blocks[second].text.casefold():
        return blocks
    # Only collapse when nothing of substance sits between the two.
    if any(not b.is_heading for b in blocks[first + 1 : second]):
        return blocks
    return [b for i, b in enumerate(blocks) if i != first]


def _assign_section_paths(blocks: list[Block]) -> None:
    """Walk the heading tree and stamp each block with its breadcrumb.

    A heading's own path includes itself, so a chunk can name its section
    without a second lookup.
    """
    stack: list[tuple[int, str]] = []
    for block in blocks:
        if block.is_heading:
            level = block.level or 2
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, block.text))
        block.section_path = tuple(text for _, text in stack)


def _id_from_filename(path: Path) -> str | None:
    match = _FILENAME_ID.match(path.stem)
    return match.group("id").upper() if match else None


def _document_hash(document_id: str, blocks: list[Block]) -> str:
    """Hash identity + structure + text.

    Deliberately excludes `source_uri`: moving the same bytes to a new container
    must not force a re-embed. Includes heading level and kind, so a
    restructured document with identical prose is correctly treated as changed.
    """
    parts = [document_id]
    parts += [f"{b.kind}:{b.level or 0}:{b.text}" for b in blocks]
    return content_hash("\n".join(parts))
