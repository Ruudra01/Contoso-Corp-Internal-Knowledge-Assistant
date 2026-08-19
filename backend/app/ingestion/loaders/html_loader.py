"""HTML loader. BeautifulSoup DOM walk -> blocks, headings from h1-h6."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag

from app.core.errors import DocumentParseError
from app.ingestion.loaders.base import register
from app.ingestion.models import Block, BlockKind, DocumentType, RawDocument

_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_DROP = ("script", "style", "nav", "footer", "noscript", "aside", "form")
_BLOCK_TAGS = _HEADINGS | {"p", "li", "table", "pre", "blockquote"}


class HtmlLoader:
    document_type = DocumentType.HTML
    extensions = (".html", ".htm", ".xhtml")

    def load(self, path: Path) -> RawDocument:
        try:
            markup = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DocumentParseError(f"cannot read html {path.name}: {exc}") from exc

        soup = BeautifulSoup(markup, "lxml")
        for tag in soup(_DROP):
            tag.decompose()

        root = soup.body or soup
        blocks = [b for tag in root.find_all(_BLOCK_TAGS) if (b := self._to_block(tag))]
        if not blocks:
            raise DocumentParseError(f"html {path.name} produced no content")

        hints: dict[str, str] = {}
        for meta in soup.find_all("meta"):
            name, content = meta.get("name"), meta.get("content")
            if name and content:
                hints[name.lower()] = content

        doc_title = soup.title.get_text(strip=True) if soup.title else None
        heading_title = next((b.text for b in blocks if b.is_heading and b.level == 1), None)
        return RawDocument(
            blocks=blocks,
            document_type=self.document_type,
            source_path=str(path),
            title_hint=heading_title or doc_title,
            metadata_hints=hints,
        )

    def _to_block(self, tag: Tag) -> Block | None:
        # Nested block tags (a <p> inside a <li>, a <li> inside a <table>) are
        # emitted by their outermost owner, so skip them here.
        if tag.name != "table" and tag.find_parent(("table", "li")) is not None:
            return None

        if tag.name == "table":
            return self._table_block(tag)

        text = " ".join(tag.get_text(" ", strip=True).split())
        if not text:
            return None
        if tag.name in _HEADINGS:
            return Block(kind=BlockKind.HEADING, text=text, level=int(tag.name[1]))
        kind = BlockKind.LIST_ITEM if tag.name == "li" else BlockKind.PARAGRAPH
        return Block(kind=kind, text=text)

    @staticmethod
    def _table_block(table: Tag) -> Block | None:
        """Render to a Markdown pipe table so all four formats express tables the
        same way downstream."""
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [
                " ".join(td.get_text(" ", strip=True).split()) for td in tr.find_all(("td", "th"))
            ]
            if any(cells):
                rows.append(cells)
        if not rows:
            return None
        width = max(len(r) for r in rows)
        lines = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in rows]
        lines.insert(1, "|" + "---|" * width)
        return Block(kind=BlockKind.TABLE, text="\n".join(lines))


register(HtmlLoader())
