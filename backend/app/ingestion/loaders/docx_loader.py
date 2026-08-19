"""DOCX loader. python-docx, headings from paragraph styles, body order
preserved by walking the document XML so tables stay in place."""

from __future__ import annotations

import re
from pathlib import Path

import docx
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.errors import DocumentParseError
from app.ingestion.loaders.base import register
from app.ingestion.models import Block, BlockKind, DocumentType, RawDocument

_HEADING_STYLE = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)
_LIST_STYLE = re.compile(r"list|bullet", re.IGNORECASE)


class DocxLoader:
    document_type = DocumentType.DOCX
    extensions = (".docx",)

    def load(self, path: Path) -> RawDocument:
        try:
            document = docx.Document(str(path))
        except Exception as exc:  # python-docx raises bare exceptions on bad zips
            raise DocumentParseError(f"cannot read docx {path.name}: {exc}") from exc

        blocks = [b for item in _iter_body(document) if (b := self._to_block(item))]
        if not blocks:
            raise DocumentParseError(f"docx {path.name} produced no content")

        core = document.core_properties
        hints = {k: v for k, v in (("title", core.title), ("subject", core.subject)) if v}
        title = next((b.text for b in blocks if b.is_heading and b.level == 1), None)
        return RawDocument(
            blocks=blocks,
            document_type=self.document_type,
            source_path=str(path),
            title_hint=title or hints.get("title"),
            metadata_hints=hints,
        )

    def _to_block(self, item: Paragraph | Table) -> Block | None:
        if isinstance(item, Table):
            return self._table_block(item)

        text = " ".join(item.text.split())
        if not text:
            return None

        style = (item.style.name if item.style is not None else "") or ""
        if level := self._heading_level(style, item):
            return Block(kind=BlockKind.HEADING, text=text, level=level)
        if _LIST_STYLE.search(style):
            return Block(kind=BlockKind.LIST_ITEM, text=text)
        return Block(kind=BlockKind.PARAGRAPH, text=text)

    @staticmethod
    def _heading_level(style: str, paragraph: Paragraph) -> int | None:
        if match := _HEADING_STYLE.match(style.strip()):
            return min(int(match.group(1)), 6)
        if style.strip().lower() in {"title", "subtitle"}:
            return 1 if style.strip().lower() == "title" else 2
        # Some authors format headings by hand: a short, fully bold, colon-free
        # line. Treat those as level 2 rather than losing the structure.
        runs = paragraph.runs
        if runs and len(paragraph.text) <= 80 and all(r.bold for r in runs if r.text.strip()):
            if not paragraph.text.rstrip().endswith((".", ":", ";")):
                return 2
        return None

    @staticmethod
    def _table_block(table: Table) -> Block | None:
        rows = [
            [" ".join(cell.text.split()) for cell in row.cells]
            for row in table.rows
        ]
        rows = [r for r in rows if any(r)]
        if not rows:
            return None
        width = max(len(r) for r in rows)
        lines = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in rows]
        lines.insert(1, "|" + "---|" * width)
        return Block(kind=BlockKind.TABLE, text="\n".join(lines))


def _iter_body(document: DocxDocument):
    """Yield paragraphs and tables in true document order.

    `document.paragraphs` and `document.tables` are separate sequences, so
    reading them independently loses the interleaving.
    """
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


register(DocxLoader())
