"""Markdown loader. markdown-it token stream -> blocks."""

from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt

from app.core.errors import DocumentParseError
from app.ingestion.loaders.base import register
from app.ingestion.models import Block, BlockKind, DocumentType, RawDocument


class MarkdownLoader:
    document_type = DocumentType.MARKDOWN
    extensions = (".md", ".markdown")

    def __init__(self) -> None:
        self._md = MarkdownIt("commonmark").enable("table")

    def load(self, path: Path) -> RawDocument:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DocumentParseError(f"cannot read markdown {path.name}: {exc}") from exc

        blocks = self._to_blocks(self._md.parse(source))
        if not blocks:
            raise DocumentParseError(f"markdown {path.name} produced no content")

        title = next((b.text for b in blocks if b.is_heading and b.level == 1), None)
        return RawDocument(
            blocks=blocks,
            document_type=self.document_type,
            source_path=str(path),
            title_hint=title,
        )

    def _to_blocks(self, tokens: list) -> list[Block]:
        blocks: list[Block] = []
        table_rows: list[str] = []
        in_table = False
        row: list[str] = []
        pending_kind: BlockKind | None = None
        heading_level: int | None = None

        for tok in tokens:
            match tok.type:
                case "heading_open":
                    pending_kind, heading_level = BlockKind.HEADING, int(tok.tag[1:])
                case "paragraph_open":
                    pending_kind = BlockKind.PARAGRAPH
                case "table_open":
                    in_table, table_rows = True, []
                case "tr_open":
                    row = []
                case "th_close" | "td_close":
                    pass
                case "tr_close":
                    if row:
                        table_rows.append("| " + " | ".join(row) + " |")
                case "table_close":
                    in_table = False
                    if table_rows:
                        blocks.append(self._table_block(table_rows))
                case "inline":
                    text = _plain(tok)
                    if in_table:
                        row.append(text)
                    elif pending_kind and text:
                        # A list item's text arrives as a paragraph inside the item.
                        blocks.append(Block(kind=pending_kind, text=text, level=heading_level))
                        pending_kind, heading_level = None, None
                case "fence" | "code_block":
                    if tok.content.strip():
                        blocks.append(
                            Block(kind=BlockKind.PARAGRAPH, text=f"```\n{tok.content.rstrip()}\n```")
                        )
                case "list_item_open":
                    pending_kind = BlockKind.LIST_ITEM
        return blocks

    @staticmethod
    def _table_block(rows: list[str]) -> Block:
        """Keep the pipe table plus its separator: it stays readable as evidence
        and survives as a single atomic block through chunking."""
        if len(rows) > 1:
            columns = rows[0].count("|") - 1
            rows = [rows[0], "|" + "---|" * columns, *rows[1:]]
        return Block(kind=BlockKind.TABLE, text="\n".join(rows))


def _plain(token) -> str:
    """Flatten an inline token to text, dropping emphasis markers."""
    if not token.children:
        return token.content.strip()
    parts = [
        child.content
        for child in token.children
        if child.type in {"text", "code_inline"}
    ]
    # Preserve the hard-break-separated `**Key:** value` metadata lines.
    return " ".join(" ".join(parts).split())


register(MarkdownLoader())
