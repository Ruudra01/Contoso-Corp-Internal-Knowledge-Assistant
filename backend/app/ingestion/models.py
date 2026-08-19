"""The common normalized document representation.

Every loader — PDF, DOCX, HTML, Markdown — produces `Block` objects and nothing
else. The normalizer and chunker therefore never branch on source format; adding
a fifth format means adding a loader, not touching downstream code.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum

_KEY_SAFE = re.compile(r"[^A-Za-z0-9_\-=]")


class DocumentType(StrEnum):
    """Supported source formats."""

    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"


class BlockKind(StrEnum):
    """Structural role of a block, format-independent."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"


@dataclass(slots=True)
class Block:
    """One structural unit of a document.

    text        already-plain text (Markdown pipe syntax retained for tables)
    level       heading depth 1-6; None for non-headings
    page_number 1-based, only formats that expose pagination (PDF) set it
    """

    kind: BlockKind
    text: str
    level: int | None = None
    page_number: int | None = None
    # Heading breadcrumb this block sits under, outermost first. Assigned by the
    # normalizer, not by loaders.
    section_path: tuple[str, ...] = ()

    @property
    def is_heading(self) -> bool:
        return self.kind is BlockKind.HEADING

    @property
    def is_atomic(self) -> bool:
        """Blocks that must never be split mid-way (a half table is unusable)."""
        return self.kind is BlockKind.TABLE


@dataclass(slots=True)
class RawDocument:
    """Direct loader output: blocks plus whatever the format volunteered."""

    blocks: list[Block]
    document_type: DocumentType
    source_path: str
    title_hint: str | None = None
    page_count: int | None = None
    # Front-matter style `Key: Value` pairs a format exposed natively
    # (DOCX core properties, HTML <meta>/<title>).
    metadata_hints: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentMetadata:
    """Document-level metadata carried onto every chunk."""

    document_id: str
    document_name: str
    document_type: DocumentType
    source_uri: str
    # Optional corpus attributes lifted from the in-document front matter.
    department: str | None = None
    category: str | None = None
    version: str | None = None
    effective_date: str | None = None
    status: str | None = None


@dataclass(slots=True)
class NormalizedDocument:
    """Format-agnostic document ready for chunking."""

    metadata: DocumentMetadata
    blocks: list[Block]
    content_hash: str
    page_count: int | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


@dataclass(slots=True)
class Chunk:
    """One indexable unit.

    text            verbatim body, what gets shown to the user as evidence
    embedding_text  body prefixed with the `doc title > H1 > H2` breadcrumb, what
                    gets embedded, so the vector encodes its own topic
    """

    document_id: str
    document_name: str
    document_type: DocumentType
    source_uri: str
    ordinal: int
    text: str
    embedding_text: str
    section: str
    section_path: tuple[str, ...]
    token_count: int
    page_number: int | None = None
    page_end: int | None = None
    content_hash: str = ""
    embedding: list[float] | None = None

    @property
    def chunk_id(self) -> str:
        """Deterministic Azure AI Search key.

        Stable across runs for the same (document, ordinal), which is what makes
        re-indexing an upsert instead of a duplicate insert. Azure keys allow
        only letters, digits, underscore, dash and equals.
        """
        return f"{_KEY_SAFE.sub('_', self.document_id)}-{self.ordinal:04d}"

    @property
    def section_path_text(self) -> str:
        return " > ".join(self.section_path)


def content_hash(text: str) -> str:
    """SHA-256 over normalized text. Drives the skip-unchanged short circuit."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
