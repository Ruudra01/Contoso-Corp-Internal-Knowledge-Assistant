"""Structure-aware chunking with token limits and overlap.

The heading tree decides boundaries; the token budget decides how much of the
tree fits in one chunk. Pipeline:

1. cut the block stream into sections at every heading;
2. pack consecutive sections into a chunk while they stay under
   `target_tokens` *and* share a common ancestor heading, so a chunk never
   straddles two unrelated parts of a document;
3. a section that alone exceeds `max_tokens` is split along *cohesive unit*
   boundaries, and consecutive splits carry `overlap_tokens` of trailing
   context so a rule and its exception are never separated silently.

A cohesive unit is a run of blocks that must not be broken apart:

* a table (half a table is not evidence);
* a consecutive run of list items (half an enumerated rule set is misleading —
  a reader cannot tell whether the conditions are conjunctive);
* a single paragraph, which is where an individual policy rule lives.

Units are only ever broken when one unit alone exceeds the budget, and when
that happens the fact is recorded in the text and the log rather than hidden.

The rule adapts to document scale without a second mode. Short policies
(a few hundred tokens) emerge as one dense chunk cited to the document; a long
handbook emerges as section-level chunks, because each of its sections already
exceeds the target.

Invariants the retrieval layer depends on:

* a chunk never crosses a top-level heading boundary, so `section_path` always
  names a real, citable location;
* every chunk carries the document id, name, type, source URI and section path,
  so it can be traced to its exact origin;
* `token_count` is measured on the text that is actually embedded, breadcrumb
  included, and is what the size limits are enforced against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.ingestion.models import Block, BlockKind, Chunk, NormalizedDocument
from app.ingestion.tokens import count_tokens, truncate_to_tokens

logger = get_logger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
# Appended to a chunk whose cohesive unit had to be broken, so a reader can see
# that the enumeration continues elsewhere.
_CONTINUES = "(continued below)"
_CONTINUED = "(continued)"


@dataclass(slots=True)
class _Section:
    """A heading and the blocks it owns, excluding nested subsections."""

    path: tuple[str, ...]
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


@dataclass(slots=True)
class _Unit:
    """A run of blocks that should not be split apart.

    `fragment` holds the text when this unit is only *part* of one block — the
    case where a single paragraph exceeded the ceiling and had to be split on
    sentence boundaries. The blocks are still carried so page numbers resolve.
    """

    blocks: list[Block]
    tokens: int
    splittable: bool
    fragment: str | None = None

    @property
    def text(self) -> str:
        if self.fragment is not None:
            return self.fragment
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def is_list(self) -> bool:
        return self.blocks[0].kind is BlockKind.LIST_ITEM


@dataclass(slots=True)
class _Group:
    """One or more consecutive sections destined for the same chunk."""

    sections: list[_Section]
    tokens: int

    @property
    def path(self) -> tuple[str, ...]:
        """Deepest heading path common to every section in the group.

        A single-section group keeps its own full path, which is what makes a
        citation precise; a merged group is cited to the shared ancestor rather
        than falsely claiming one subsection.
        """
        paths = [s.path for s in self.sections]
        return _common_prefix(paths) if len(paths) > 1 else paths[0]

    @property
    def blocks(self) -> list[Block]:
        return [b for s in self.sections for b in s.blocks]

    def text(self, base_depth: int) -> str:
        """Body text. When the group spans several sections, each section's own
        heading is written into the body so the hierarchy survives in the text
        the model reads."""
        if len(self.sections) == 1:
            return self.sections[0].text
        parts: list[str] = []
        for section in self.sections:
            heading = " > ".join(section.path[base_depth:])
            parts.append(f"{heading}\n{section.text}" if heading else section.text)
        return "\n\n".join(parts)


class StructureAwareChunker:
    """Structure-aware chunker with a configurable ceiling and overlap.

    target_tokens   preferred chunk size; sections pack up to this
    max_tokens      hard ceiling; a section above this is split
    overlap_tokens  trailing context repeated into the next split of a section
    """

    def __init__(
        self,
        *,
        target_tokens: int = 800,
        overlap_tokens: int = 120,
        max_tokens: int = 1200,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must not be negative")
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        if max_tokens < target_tokens:
            raise ValueError("max_tokens must be >= target_tokens")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.max_tokens = max_tokens

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        name = document.metadata.document_name
        sections = self._sections(document.blocks)
        bodies: list[tuple[tuple[str, ...], str, list[Block]]] = []
        for group in self._pack(sections, document_name=name):
            bodies.extend(self._bodies_for(group, document_name=name))

        chunks = [
            self._build_chunk(document, ordinal, path, text, blocks)
            for ordinal, (path, text, blocks) in enumerate(bodies)
        ]
        logger.debug(
            "chunked document",
            extra={
                "document_id": document.metadata.document_id,
                "sections": len(sections),
                "chunks": len(chunks),
                "tokens": sum(c.token_count for c in chunks),
            },
        )
        return chunks

    # -- sectioning ---------------------------------------------------------

    @staticmethod
    def _sections(blocks: list[Block]) -> list[_Section]:
        """Cut the block stream at every heading.

        Sections with no body blocks are dropped: their heading is not lost, it
        survives in the `section_path` of every descendant. A document with no
        headings at all yields exactly one section with an empty path, which
        `_build_chunk` resolves to the document name.
        """
        sections: list[_Section] = []
        current = _Section(path=())
        for block in blocks:
            if block.is_heading:
                if current.blocks:
                    sections.append(current)
                current = _Section(path=block.section_path)
                continue
            current.blocks.append(block)
        if current.blocks:
            sections.append(current)
        return sections

    # -- budgets ------------------------------------------------------------

    def _overhead(self, document_name: str, path: tuple[str, ...]) -> int:
        """Tokens the breadcrumb prefix will add to the embedded text.

        Counted against the budget so the size limits hold for the text that is
        actually embedded, not just the body.
        """
        breadcrumb = self._breadcrumb(document_name, path)
        return count_tokens(f"{breadcrumb}\n\n") if breadcrumb else 0

    # -- packing sections ---------------------------------------------------

    def _pack(self, sections: list[_Section], *, document_name: str) -> list[_Group]:
        """Greedily group consecutive related sections up to the token target."""
        groups: list[_Group] = []
        for section in sections:
            tokens = count_tokens(section.text)
            if groups and self._can_join(groups[-1], section, tokens, document_name):
                groups[-1].sections.append(section)
                groups[-1].tokens += tokens
                continue
            groups.append(_Group(sections=[section], tokens=tokens))
        return groups

    def _can_join(
        self, group: _Group, section: _Section, tokens: int, document_name: str
    ) -> bool:
        # Require a shared ancestor heading. Depth 0 means the two sections sit
        # under different top-level headings (or one is document preamble), and
        # merging them would produce an uncitable chunk.
        merged_paths = [s.path for s in group.sections] + [section.path]
        if not _common_prefix(merged_paths):
            return False
        overhead = self._overhead(document_name, _common_prefix(merged_paths))
        return group.tokens + tokens + overhead <= self.target_tokens

    # -- splitting one oversized section ------------------------------------

    def _bodies_for(
        self, group: _Group, *, document_name: str
    ) -> list[tuple[tuple[str, ...], str, list[Block]]]:
        """Emit chunk bodies for a group, splitting if it breaches the ceiling."""
        path = group.path
        overhead = self._overhead(document_name, path)
        text = group.text(base_depth=len(path))

        if group.tokens + overhead <= self.max_tokens:
            return [(path, text, group.blocks)]

        # Only a single-section group can reach here: `_can_join` never grows a
        # group past `target_tokens`, and `target_tokens <= max_tokens`. So the
        # section's own heading path is the whole story and no interleaved
        # headings need rebuilding.
        # Budgets are for the *body*: the breadcrumb prefix is added later, and a
        # split unit also gets a continuation marker appended. Both are reserved
        # up front so the ceiling holds for the text that is finally embedded.
        reserve = count_tokens(f"\n{_CONTINUES}")
        target = max(1, self.target_tokens - overhead)
        ceiling = max(1, self.max_tokens - overhead)
        units = self._cohesive_units(
            group.blocks, ceiling=max(1, ceiling - reserve), path=path
        )
        return [
            (path, body, blocks)
            for body, blocks in self._pack_units(units, max(1, target - reserve), ceiling)
        ]

    def _cohesive_units(
        self, blocks: list[Block], *, ceiling: int, path: tuple[str, ...]
    ) -> list[_Unit]:
        """Group blocks into runs that must not be broken apart.

        Consecutive list items become one unit, so an enumerated rule set is not
        cut in half. A unit that alone exceeds the ceiling is broken here, and
        only here, with the break recorded.
        """
        units: list[_Unit] = []
        run: list[Block] = []

        def flush_run() -> None:
            nonlocal run
            if run:
                units.extend(self._bound_unit(run, ceiling=ceiling, path=path))
                run = []

        for block in blocks:
            if block.kind is BlockKind.LIST_ITEM:
                run.append(block)
                continue
            flush_run()
            units.extend(self._bound_unit([block], ceiling=ceiling, path=path))
        flush_run()
        return units

    def _bound_unit(
        self, blocks: list[Block], *, ceiling: int, path: tuple[str, ...]
    ) -> list[_Unit]:
        """Return one unit, or several if this run alone exceeds the ceiling."""
        text = "\n\n".join(b.text for b in blocks)
        tokens = count_tokens(text)
        if tokens <= ceiling:
            return [_Unit(blocks=list(blocks), tokens=tokens, splittable=False)]

        section = path[-1] if path else ""

        # An oversized list: break at item boundaries, never mid-item.
        if len(blocks) > 1 and blocks[0].kind is BlockKind.LIST_ITEM:
            logger.warning(
                "list run exceeds max_tokens, breaking at item boundaries",
                extra={"section": section, "tokens": tokens, "items": len(blocks)},
            )
            return [
                _Unit(blocks=part, tokens=count_tokens("\n\n".join(b.text for b in part)), splittable=True)
                for part in self._split_items(blocks, ceiling)
            ]

        block = blocks[0]
        # A table is atomic even when oversized: emitting half of it would be
        # actively misleading, so it goes whole and the embedder truncates if the
        # model's request limit is the binding constraint.
        if block.is_atomic:
            logger.warning(
                "atomic block exceeds max_tokens, emitting whole",
                extra={"section": section, "tokens": tokens},
            )
            return [_Unit(blocks=[block], tokens=tokens, splittable=False)]

        logger.warning(
            "paragraph exceeds max_tokens, splitting on sentence boundaries",
            extra={"section": section, "tokens": tokens},
        )
        return [
            _Unit(blocks=[block], tokens=count_tokens(part), splittable=True, fragment=part)
            for part in self._split_sentences(block.text, ceiling)
        ]

    def _split_items(self, blocks: list[Block], ceiling: int) -> list[list[Block]]:
        """Break a list run into groups of whole items.

        The budget is checked against the assembled text rather than a sum of
        per-item counts: joining changes tokenization slightly, and a sum-based
        check drifts over the ceiling by a token or two.
        """
        parts: list[list[Block]] = []
        current: list[Block] = []
        for block in blocks:
            if current and count_tokens("\n\n".join(b.text for b in [*current, block])) > ceiling:
                parts.append(current)
                current = []
            current.append(block)
        if current:
            parts.append(current)
        return parts

    def _split_sentences(self, text: str, ceiling: int) -> list[str]:
        """Sentence-aligned split of one oversized paragraph."""
        sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
        parts: list[str] = []
        current: list[str] = []
        for sentence in sentences:
            if current and count_tokens(" ".join([*current, sentence])) > ceiling:
                parts.append(" ".join(current))
                current = []
            current.append(sentence)
        if current:
            parts.append(" ".join(current))
        return parts or [text]

    def _pack_units(
        self, units: list[_Unit], target: int, ceiling: int
    ) -> list[tuple[str, list[Block]]]:
        """Pack units into chunk bodies, carrying overlap between consecutive ones.

        Overlap is trailing text from the previous chunk repeated at the head of
        the next. It is context, not new content, so the emitted block list stays
        the chunk's own units — page numbers therefore describe the chunk's own
        material, not the borrowed prefix.
        """
        packed: list[tuple[str, list[Block]]] = []
        buffer: list[_Unit] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            body = "\n\n".join(u.text for u in buffer)
            # Mark a break that fell inside a cohesive unit.
            if buffer[-1].splittable:
                body = f"{body}\n{_CONTINUES}"
            blocks = [b for u in buffer for b in u.blocks]

            if packed and self.overlap_tokens:
                # Headroom is measured on the assembled body, not the unit token
                # sum: the joins and the continuation marker cost tokens too, and
                # overlap must never push a chunk past the ceiling.
                headroom = ceiling - count_tokens(body)
                if prefix := self._overlap_prefix(packed[-1][0], headroom):
                    candidate = f"{prefix}\n\n{body}"
                    # Final exact check: joining costs a token or two of its own,
                    # so the assembled result is what gets validated.
                    if count_tokens(candidate) <= ceiling:
                        body = candidate
            packed.append((body, blocks))
            buffer, buffer_tokens = [], 0

        for unit in units:
            if buffer and count_tokens("\n\n".join(u.text for u in [*buffer, unit])) > target:
                flush()
            buffer.append(unit)
            buffer_tokens += unit.tokens
        flush()
        return packed

    def _overlap_prefix(self, previous_body: str, headroom: int) -> str:
        """Trailing sentences of the previous chunk, up to `overlap_tokens`.

        Capped by the headroom left under the ceiling so overlap can never push
        a chunk over its configured maximum.
        """
        budget = min(self.overlap_tokens, max(0, headroom))
        if budget <= 0:
            return ""
        body = previous_body.removesuffix(_CONTINUES).rstrip()
        sentences = [s for s in _SENTENCE_END.split(body) if s.strip()]
        tail: list[str] = []
        tokens = 0
        for sentence in reversed(sentences):
            sentence_tokens = count_tokens(sentence)
            if tail and tokens + sentence_tokens > budget:
                break
            if not tail and sentence_tokens > budget:
                # A single trailing sentence larger than the budget: skip rather
                # than blow the ceiling.
                return ""
            tail.insert(0, sentence)
            tokens += sentence_tokens
        if not tail:
            return ""
        return f"{_CONTINUED} {' '.join(tail)}"

    # -- chunk assembly -----------------------------------------------------

    def _build_chunk(
        self,
        document: NormalizedDocument,
        ordinal: int,
        path: tuple[str, ...],
        text: str,
        blocks: list[Block],
    ) -> Chunk:
        meta = document.metadata
        pages = sorted({b.page_number for b in blocks if b.page_number is not None})
        breadcrumb = self._breadcrumb(meta.document_name, path)
        embedding_text = f"{breadcrumb}\n\n{text}" if breadcrumb else text
        # A document with no headings still has to be traceable to a section, so
        # the document itself is the section of record.
        section_path = path or (meta.document_name,)
        return Chunk(
            document_id=meta.document_id,
            document_name=meta.document_name,
            document_type=meta.document_type,
            source_uri=meta.source_uri,
            ordinal=ordinal,
            text=text,
            embedding_text=embedding_text,
            section=path[-1] if path else meta.document_name,
            section_path=section_path,
            token_count=count_tokens(embedding_text),
            page_number=pages[0] if pages else None,
            page_end=pages[-1] if pages else None,
            content_hash=document.content_hash,
        )

    def _breadcrumb(self, document_name: str, path: tuple[str, ...]) -> str:
        """`doc title > H1 > H2`, dropping the H1 when it just repeats the title.

        Capped at half the chunk ceiling. A pathologically long title would
        otherwise consume the whole budget and make `max_tokens` unsatisfiable —
        the breadcrumb is an embedding aid, so it yields to the evidence.
        """
        parts = [document_name]
        parts += [p for p in path if p.casefold() != document_name.casefold()]
        breadcrumb = " > ".join(parts)

        budget = max(1, self.max_tokens // 2)
        if count_tokens(breadcrumb) <= budget:
            return breadcrumb
        logger.warning(
            "breadcrumb exceeds half the chunk ceiling, truncating",
            extra={"budget_tokens": budget, "section_path": breadcrumb[:80]},
        )
        return truncate_to_tokens(breadcrumb, budget)


def _common_prefix(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    if not paths:
        return ()
    prefix = paths[0]
    for path in paths[1:]:
        limit = min(len(prefix), len(path))
        cut = limit
        for i in range(limit):
            if prefix[i] != path[i]:
                cut = i
                break
        prefix = prefix[:cut]
        if not prefix:
            break
    return prefix
