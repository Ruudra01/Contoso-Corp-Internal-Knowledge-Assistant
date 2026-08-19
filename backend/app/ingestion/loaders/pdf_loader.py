"""PDF loader. PyMuPDF, keeps page numbers, recovers structure from layout.

PDF carries no semantic markup, so both headings and paragraph boundaries have
to be inferred.

Headings, in order of trust:
1. the embedded outline / bookmarks (`get_toc`) when the producer wrote one;
2. typography — a line materially larger than the document's body size, or bold
   and short. Level is the rank of its size among heading-sized lines, so an
   18pt run outranks a 14pt one.

Paragraphs: from vertical gaps. A wrapped continuation line sits flush against
its predecessor, while a real paragraph break leaves visible leading. Terminal
punctuation is not reliable here — policy text is full of `Accrual:` style
inline labels and abbreviations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.core.errors import DocumentParseError
from app.ingestion.loaders.base import register
from app.ingestion.models import Block, BlockKind, DocumentType, RawDocument

# A line must exceed body size by this ratio to count as a heading on size alone.
_HEADING_SIZE_RATIO = 1.15
_MAX_HEADING_CHARS = 120
# Vertical gap above this fraction of the line's font size starts a new
# paragraph. Wrapped lines measure ~0; single-spaced breaks measure ~0.7x.
_PARAGRAPH_GAP_RATIO = 0.45
_BULLET_CHARS = "•●▪◦-–*"


@dataclass(slots=True)
class _Line:
    text: str
    size: float
    bold: bool
    page_number: int
    top: float
    bottom: float


class PdfLoader:
    document_type = DocumentType.PDF
    extensions = (".pdf",)

    def load(self, path: Path) -> RawDocument:
        try:
            document = pymupdf.open(str(path))
        except Exception as exc:
            raise DocumentParseError(f"cannot open pdf {path.name}: {exc}") from exc

        try:
            if document.is_encrypted and not document.authenticate(""):
                raise DocumentParseError(f"pdf {path.name} is password protected")
            lines = self._extract_lines(document)
            page_count = document.page_count
            outline_titles = {
                " ".join(entry[1].split()) for entry in document.get_toc() if len(entry) > 1
            }
        finally:
            document.close()

        if not lines:
            raise DocumentParseError(
                f"pdf {path.name} yielded no text layer (likely a scan; OCR is out of scope)"
            )

        blocks = self._to_blocks(lines, outline_titles)
        title = next((b.text for b in blocks if b.is_heading and b.level == 1), None)
        return RawDocument(
            blocks=blocks,
            document_type=self.document_type,
            source_path=str(path),
            title_hint=title,
            page_count=page_count,
        )

    @staticmethod
    def _extract_lines(document: pymupdf.Document) -> list[_Line]:
        lines: list[_Line] = []
        for page_index, page in enumerate(document, start=1):
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type"):  # 1 == image
                    continue
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = " ".join("".join(s["text"] for s in spans).split())
                    if not text:
                        continue
                    dominant = max(spans, key=lambda s: len(s["text"]))
                    _, top, _, bottom = line["bbox"]
                    lines.append(
                        _Line(
                            text=text,
                            size=round(float(dominant.get("size", 0.0)), 1),
                            bold="bold" in str(dominant.get("font", "")).lower(),
                            page_number=page_index,
                            top=float(top),
                            bottom=float(bottom),
                        )
                    )
        return lines

    def _to_blocks(self, lines: list[_Line], outline_titles: set[str]) -> list[Block]:
        body_size = self._body_size(lines)
        heading_sizes = sorted(
            {ln.size for ln in lines if self._is_heading(ln, body_size, outline_titles)},
            reverse=True,
        )
        # Largest heading size -> level 1, next -> level 2, capped at 6.
        level_of = {size: min(rank, 6) for rank, size in enumerate(heading_sizes, start=1)}

        blocks: list[Block] = []
        buffer: list[str] = []
        buffer_kind = BlockKind.PARAGRAPH
        buffer_page: int | None = None
        previous: _Line | None = None

        def flush() -> None:
            nonlocal buffer, buffer_page, buffer_kind
            if buffer:
                text = " ".join(" ".join(buffer).split())
                if text:
                    blocks.append(Block(kind=buffer_kind, text=text, page_number=buffer_page))
            buffer, buffer_page, buffer_kind = [], None, BlockKind.PARAGRAPH

        for line in lines:
            if self._is_heading(line, body_size, outline_titles):
                flush()
                blocks.append(
                    Block(
                        kind=BlockKind.HEADING,
                        text=line.text,
                        level=level_of.get(line.size, 2),
                        page_number=line.page_number,
                    )
                )
                previous = line
                continue

            stripped = line.text.lstrip()
            is_bullet = bool(stripped) and stripped[0] in _BULLET_CHARS
            if is_bullet or self._starts_new_paragraph(previous, line):
                flush()

            if is_bullet:
                buffer_kind = BlockKind.LIST_ITEM
                line_text = stripped.lstrip(_BULLET_CHARS).strip()
            else:
                line_text = line.text

            if buffer_page is None:
                buffer_page = line.page_number
            buffer.append(line_text)
            previous = line

        flush()
        return blocks

    @staticmethod
    def _starts_new_paragraph(previous: _Line | None, line: _Line) -> bool:
        if previous is None:
            return False
        if previous.page_number != line.page_number:
            return True
        threshold = max(line.size, 1.0) * _PARAGRAPH_GAP_RATIO
        return (line.top - previous.bottom) > threshold

    @staticmethod
    def _body_size(lines: list[_Line]) -> float:
        """Most common font size weighted by character count = the body text."""
        counter: Counter[float] = Counter()
        for line in lines:
            counter[line.size] += len(line.text)
        return counter.most_common(1)[0][0] if counter else 0.0

    @staticmethod
    def _is_heading(line: _Line, body_size: float, outline_titles: set[str]) -> bool:
        if len(line.text) > _MAX_HEADING_CHARS:
            return False
        if line.text in outline_titles:
            return True
        if body_size and line.size >= body_size * _HEADING_SIZE_RATIO:
            return True
        # Bold, short, and not a sentence: a hand-formatted heading.
        return line.bold and len(line.text) <= 80 and not line.text.rstrip().endswith((".", ","))


register(PdfLoader())
