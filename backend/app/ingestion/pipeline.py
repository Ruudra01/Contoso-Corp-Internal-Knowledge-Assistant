"""Ingestion orchestration: Document -> Parse -> Normalize -> Chunk -> Embed -> Index.

Idempotency
-----------
Loading and normalizing are cheap and always run, because the content hash is
derived from normalized content. Everything expensive downstream — embedding and
indexing — is skipped when the hash already in the index matches. `force=True`
bypasses the gate for a deliberate rebuild.

Re-index safety
---------------
Chunk keys are deterministic, so writes are upserts and never duplicate. After
upserting, keys the document no longer produces are deleted, so a document that
loses a section does not leave stale chunks answering queries. A document that
fails mid-run leaves the previous version intact rather than a half-written one.

Errors
------
One bad document does not stop the run. Each failure is logged with its document
and recorded in the report; the process exit code reflects whether any occurred.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Sequence

from app.core.config import IngestionSettings, get_settings
from app.core.errors import IngestionError
from app.core.logging import get_logger
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.embedder import Embedder, build_embedder
from app.search import SearchStore, build_search_store
from app.ingestion.loaders import loader_for, supported_extensions
from app.ingestion.models import Chunk, NormalizedDocument
from app.ingestion.normalizer import normalize

logger = get_logger(__name__)

# Repository furniture that lives alongside the corpus but is not corpus content.
_SKIP_NAMES = {".DS_Store", "Thumbs.db", "README.md", "readme.md"}


class Outcome(StrEnum):
    """Mirrors `document_ingestions.outcome` in the data model."""

    INDEXED = "indexed"
    SKIPPED_UNCHANGED = "skipped_unchanged"
    FAILED = "failed"


@dataclass(slots=True)
class DocumentResult:
    source_path: str
    outcome: Outcome
    document_id: str | None = None
    document_name: str | None = None
    document_type: str | None = None
    chunks_written: int = 0
    chunks_deleted: int = 0
    embedding_tokens: int = 0
    duration_ms: int = 0
    content_hash: str | None = None
    error: str | None = None


@dataclass(slots=True)
class IngestionReport:
    """Mirrors the `ingestion_runs` row this would persist."""

    results: list[DocumentResult] = field(default_factory=list)
    duration_ms: int = 0

    @property
    def docs_total(self) -> int:
        return len(self.results)

    @property
    def docs_indexed(self) -> int:
        return sum(r.outcome is Outcome.INDEXED for r in self.results)

    @property
    def docs_skipped(self) -> int:
        return sum(r.outcome is Outcome.SKIPPED_UNCHANGED for r in self.results)

    @property
    def docs_failed(self) -> int:
        return sum(r.outcome is Outcome.FAILED for r in self.results)

    @property
    def chunks_written(self) -> int:
        return sum(r.chunks_written for r in self.results)

    @property
    def chunks_deleted(self) -> int:
        return sum(r.chunks_deleted for r in self.results)

    @property
    def embedding_tokens(self) -> int:
        return sum(r.embedding_tokens for r in self.results)

    @property
    def status(self) -> str:
        return "failed" if self.docs_failed else "succeeded"

    def summary(self) -> dict[str, object]:
        return {
            "status": self.status,
            "docs_total": self.docs_total,
            "docs_indexed": self.docs_indexed,
            "docs_skipped": self.docs_skipped,
            "docs_failed": self.docs_failed,
            "chunks_written": self.chunks_written,
            "chunks_deleted": self.chunks_deleted,
            "embedding_tokens": self.embedding_tokens,
            "duration_ms": self.duration_ms,
        }


def discover_documents(root: Path) -> list[Path]:
    """Every supported file under `root`, sorted for a stable run order."""
    if root.is_file():
        return [root]
    if not root.exists():
        raise IngestionError(f"corpus root does not exist: {root}")
    extensions = set(supported_extensions())
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
        and path.name not in _SKIP_NAMES
        and not path.name.startswith(".")
    )


class IngestionPipeline:
    def __init__(
        self,
        *,
        settings: IngestionSettings | None = None,
        embedder: Embedder | None = None,
        indexer: SearchStore | None = None,
        chunker: StructureAwareChunker | None = None,
        offline: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        chunking = self.settings.chunking
        self.chunker = chunker or StructureAwareChunker(
            target_tokens=chunking.target_tokens,
            overlap_tokens=chunking.overlap_tokens,
            max_tokens=chunking.max_tokens,
        )
        self.embedder = embedder or build_embedder(self.settings.openai, offline=offline)
        self.indexer = indexer or build_search_store(
            self.settings.search,
            vector_dimensions=self.embedder.dimensions,
            offline=offline,
        )

    def run(
        self,
        paths: Iterable[Path] | None = None,
        *,
        force: bool = False,
        dry_run: bool = False,
        update_index: bool = False,
    ) -> IngestionReport:
        started = time.perf_counter()
        documents = list(paths) if paths is not None else discover_documents(self.settings.corpus_root)
        report = IngestionReport()

        logger.info(
            "ingestion run starting",
            extra={
                "documents": len(documents),
                "force": force,
                "dry_run": dry_run,
                "index": self.settings.search.index_name,
                "target_tokens": self.settings.chunking.target_tokens,
                "overlap_tokens": self.settings.chunking.overlap_tokens,
            },
        )

        if not dry_run:
            self.indexer.ensure_index(allow_update=update_index)
        known_hashes = {} if force else self._known_hashes(dry_run=dry_run)

        for path in documents:
            report.results.append(
                self._ingest_one(path, known_hashes=known_hashes, force=force, dry_run=dry_run)
            )

        report.duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info("ingestion run finished", extra=report.summary())
        return report

    def _known_hashes(self, *, dry_run: bool) -> dict[str, str]:
        try:
            return self.indexer.existing_hashes()
        except IngestionError as exc:
            # A failure to read prior state must not silently cause a full
            # re-embed of the corpus, so the run stops here.
            if dry_run:
                logger.warning("could not read existing hashes in dry run", extra={"error": str(exc)})
                return {}
            raise

    def _ingest_one(
        self,
        path: Path,
        *,
        known_hashes: dict[str, str],
        force: bool,
        dry_run: bool,
    ) -> DocumentResult:
        started = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        try:
            document = self._parse_and_normalize(path)
        except IngestionError as exc:
            logger.error("document failed to parse", extra={"source": str(path), "error": str(exc)})
            return DocumentResult(
                source_path=str(path), outcome=Outcome.FAILED, error=str(exc), duration_ms=elapsed_ms()
            )
        except Exception as exc:  # a loader dependency misbehaving is still one document's problem
            logger.exception("unexpected error normalizing document", extra={"source": str(path)})
            return DocumentResult(
                source_path=str(path),
                outcome=Outcome.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed_ms(),
            )

        meta = document.metadata
        base = DocumentResult(
            source_path=str(path),
            outcome=Outcome.SKIPPED_UNCHANGED,
            document_id=meta.document_id,
            document_name=meta.document_name,
            document_type=str(meta.document_type),
            content_hash=document.content_hash,
        )

        if not force and known_hashes.get(meta.document_id) == document.content_hash:
            base.duration_ms = elapsed_ms()
            logger.info(
                "skipping unchanged document",
                extra={"document_id": meta.document_id, "content_hash": document.content_hash[:12]},
            )
            return base

        try:
            chunks = self.chunker.chunk(document)
            if not chunks:
                raise IngestionError("document produced no chunks")

            if dry_run:
                base.outcome = Outcome.INDEXED
                base.chunks_written = len(chunks)
                base.duration_ms = elapsed_ms()
                logger.info(
                    "dry run: would index document",
                    extra={"document_id": meta.document_id, "chunks": len(chunks)},
                )
                return base

            tokens_before = getattr(self.embedder, "total_tokens", 0)
            self._embed(chunks)
            written = self._upsert(meta.document_id, chunks)
            deleted = self._prune(meta.document_id, chunks)

            base.outcome = Outcome.INDEXED
            base.chunks_written = written
            base.chunks_deleted = deleted
            base.embedding_tokens = max(0, getattr(self.embedder, "total_tokens", 0) - tokens_before)
            base.duration_ms = elapsed_ms()
            logger.info(
                "document indexed",
                extra={
                    "document_id": meta.document_id,
                    "document_type": str(meta.document_type),
                    "chunks_written": written,
                    "chunks_deleted": deleted,
                    "embedding_tokens": base.embedding_tokens,
                    "duration_ms": base.duration_ms,
                },
            )
            return base

        except Exception as exc:
            logger.error(
                "document failed to index",
                extra={"document_id": meta.document_id, "error": str(exc)},
                exc_info=not isinstance(exc, IngestionError),
            )
            base.outcome = Outcome.FAILED
            base.error = str(exc)
            base.duration_ms = elapsed_ms()
            return base

    def _parse_and_normalize(self, path: Path) -> NormalizedDocument:
        raw = loader_for(path).load(path)
        return normalize(raw, source_uri=self._source_uri(path))

    def _source_uri(self, path: Path) -> str:
        """Where a citation should point. Blob URL when configured, else the
        local file, so a developer run still yields a resolvable link."""
        base = self.settings.source_uri_base.rstrip("/")
        return f"{base}/{path.name}" if base else path.resolve().as_uri()

    def _embed(self, chunks: Sequence[Chunk]) -> None:
        vectors = self.embedder.embed([c.embedding_text for c in chunks])
        if len(vectors) != len(chunks):
            raise IngestionError(
                f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector

    def _upsert(self, document_id: str, chunks: Sequence[Chunk]) -> int:
        """Write chunks, treating any rejected document as a document failure.

        A partially indexed document is worse than a skipped one: it would answer
        queries from half its sections while reporting success. So a partial
        failure raises, the document is recorded as failed, and the previous
        version stays in the index untouched.
        """
        result = self.indexer.upsert(chunks)
        if result.failed:
            first = result.failed[0]
            raise IngestionError(
                f"{result.failed_count}/{len(chunks)} chunks rejected by the index; "
                f"first: {first.chunk_id} {first.error_message}"
            )
        return result.succeeded_count

    def _prune(self, document_id: str, chunks: Sequence[Chunk]) -> int:
        """Remove chunks this document produced on a previous, longer run."""
        current = {c.chunk_id for c in chunks}
        stale = self.indexer.chunk_ids_for(document_id) - current
        if not stale:
            return 0
        logger.info(
            "deleting stale chunks",
            extra={"document_id": document_id, "count": len(stale)},
        )
        result = self.indexer.delete(sorted(stale))
        if result.failed:
            # Stale chunks left behind are a correctness problem: they keep
            # answering queries from withdrawn content.
            logger.error(
                "could not delete every stale chunk",
                extra={"document_id": document_id, **result.summary()},
            )
        return result.succeeded_count
