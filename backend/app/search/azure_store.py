"""Azure AI Search implementation of `SearchStore`.

The only module in the application that imports the Azure search SDK.

Authentication: an admin key from configuration when present, otherwise
`DefaultAzureCredential` (Managed Identity in Container Apps). No credential is
read from source, and none is ever logged — log records carry the endpoint and
index name only.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from app.core.config import AzureSearchSettings
from app.core.errors import ConfigurationError, IndexingError
from app.core.logging import get_logger
from app.ingestion.models import Chunk
from app.search.models import (
    DocumentIndexError,
    IndexHealth,
    IndexingResult,
    SearchHit,
    SearchMode,
)
from app.search.schema import (
    FIELD_CHUNK_ID,
    FIELD_CONTENT_HASH,
    FIELD_DOCUMENT_ID,
    FIELD_DOCUMENT_TYPE,
    FIELD_EMBEDDING,
    FIELD_ORDINAL,
    RETRIEVAL_FIELDS,
    SEMANTIC_CONFIG_NAME,
    build_index,
    chunk_to_document,
    document_to_hit,
    vector_dimensions_of,
)

logger = get_logger(__name__)

# `search(top=...)` upper bound per request; paging beyond this needs skip.
_MAX_TOP = 1000
# Enough for the 24-document corpus and comfortably within service limits.
_MAX_CHUNK_LISTING = 10_000


class AzureSearchStore:
    """Chunk index backed by Azure AI Search."""

    def __init__(self, settings: AzureSearchSettings, *, vector_dimensions: int) -> None:
        if not settings.endpoint:
            raise ConfigurationError(
                "AZURE_SEARCH_ENDPOINT is not set; export it or use InMemorySearchStore "
                "for offline runs"
            )
        if vector_dimensions <= 0:
            raise ConfigurationError("vector_dimensions must be positive")
        self._settings = settings
        self.index_name = settings.index_name
        self.vector_dimensions = vector_dimensions
        self._credential = self._build_credential(settings)

    def __repr__(self) -> str:  # keeps a stray log or traceback secret-free
        return f"AzureSearchStore(endpoint={self._settings.endpoint!r}, index={self.index_name!r})"

    # -- clients ------------------------------------------------------------

    @staticmethod
    def _build_credential(settings: AzureSearchSettings):
        if settings.api_key:
            from azure.core.credentials import AzureKeyCredential

            logger.info(
                "search client authenticating with API key from configuration",
                extra={"endpoint": settings.endpoint, "index": settings.index_name},
            )
            return AzureKeyCredential(settings.api_key)

        from azure.identity import DefaultAzureCredential

        logger.info(
            "search client authenticating with DefaultAzureCredential",
            extra={"endpoint": settings.endpoint, "index": settings.index_name},
        )
        return DefaultAzureCredential()

    def _search_client(self):
        from azure.search.documents import SearchClient

        return SearchClient(
            endpoint=self._settings.endpoint,
            index_name=self.index_name,
            credential=self._credential,
        )

    def _index_client(self):
        from azure.search.documents.indexes import SearchIndexClient

        return SearchIndexClient(endpoint=self._settings.endpoint, credential=self._credential)

    # -- schema -------------------------------------------------------------

    def ensure_index(self, *, allow_update: bool = False) -> bool:
        """Create the index, or update it in place when explicitly allowed.

        A change in vector width is refused: Azure AI Search cannot re-dimension
        a populated vector field, so the honest answer is a new index and a
        re-ingest, not a silent no-op.
        """
        from azure.core.exceptions import ResourceNotFoundError

        client = self._index_client()
        definition = build_index(self.index_name, self.vector_dimensions)

        try:
            existing = client.get_index(self.index_name)
        except ResourceNotFoundError:
            logger.info("creating search index", extra={"index": self.index_name})
            try:
                client.create_index(definition)
            except Exception as exc:
                raise IndexingError(f"could not create index {self.index_name}: {exc}") from exc
            return True
        except Exception as exc:
            raise IndexingError(f"could not read index {self.index_name}: {exc}") from exc

        current_dimensions = vector_dimensions_of(existing)
        if current_dimensions not in (None, self.vector_dimensions):
            raise IndexingError(
                f"index {self.index_name} has vector width {current_dimensions}, "
                f"configuration expects {self.vector_dimensions}. Azure AI Search cannot "
                "re-dimension a vector field: create a new index (bump AZURE_SEARCH_INDEX_NAME) "
                "and re-ingest."
            )

        if not allow_update:
            logger.info("search index present, leaving schema untouched", extra={"index": self.index_name})
            return False

        logger.info("updating search index schema", extra={"index": self.index_name})
        try:
            client.create_or_update_index(definition)
        except Exception as exc:
            raise IndexingError(f"could not update index {self.index_name}: {exc}") from exc
        return True

    def delete_index(self) -> None:
        """Drop the index. Used by re-index-from-scratch flows and test cleanup."""
        try:
            self._index_client().delete_index(self.index_name)
        except Exception as exc:
            raise IndexingError(f"could not delete index {self.index_name}: {exc}") from exc
        logger.warning("search index deleted", extra={"index": self.index_name})

    def health(self) -> IndexHealth:
        """Connectivity and readiness probe. Returns a verdict, never raises."""
        from azure.core.exceptions import ResourceNotFoundError

        try:
            client = self._index_client()
            index = client.get_index(self.index_name)
        except ResourceNotFoundError:
            # The service answered, so it is reachable; the index just is not there.
            return IndexHealth(
                reachable=True,
                index_name=self.index_name,
                index_exists=False,
                error="index does not exist",
            )
        except Exception as exc:
            logger.error(
                "search service unreachable",
                extra={"index": self.index_name, "error": str(exc)},
            )
            return IndexHealth(reachable=False, index_name=self.index_name, error=str(exc))

        document_count: int | None = None
        try:
            statistics = client.get_index_statistics(self.index_name)
            document_count = (statistics or {}).get("document_count")
        except Exception as exc:
            # Statistics are informational; their absence is not unhealthy.
            logger.warning(
                "could not read index statistics",
                extra={"index": self.index_name, "error": str(exc)},
            )

        return IndexHealth(
            reachable=True,
            index_name=self.index_name,
            index_exists=True,
            document_count=document_count,
            vector_dimensions=vector_dimensions_of(index),
        )

    # -- writes -------------------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk]) -> IndexingResult:
        """Insert or replace chunks, batched.

        Keys are deterministic, so this is an upsert and re-running it is a
        no-op rather than a duplicate insert. A batch the service partially
        rejects yields a result carrying both sides; the caller decides.
        """
        if not chunks:
            return IndexingResult()

        client = self._search_client()
        result = IndexingResult()
        batch_size = self._settings.upload_batch_size

        for start in range(0, len(chunks), batch_size):
            window = chunks[start : start + batch_size]
            documents = [chunk_to_document(c) for c in window]
            try:
                responses = client.merge_or_upload_documents(documents=documents)
            except Exception as exc:
                # A transport-level failure rejects the whole batch; attribute it
                # to every key so nothing is silently reported as written.
                logger.error(
                    "index batch failed outright",
                    extra={
                        "index": self.index_name,
                        "batch_start": start,
                        "batch_size": len(window),
                        "error": str(exc),
                    },
                )
                result.failed.extend(
                    DocumentIndexError(chunk_id=c.chunk_id, status_code=None, error_message=str(exc))
                    for c in window
                )
                continue
            result += self._collect(responses, operation="upsert")

        if result.failed:
            logger.error(
                "index upsert completed with failures",
                extra={"index": self.index_name, **result.summary()},
            )
        else:
            logger.info(
                "index upsert complete",
                extra={"index": self.index_name, "documents": result.succeeded_count},
            )
        return result

    def delete(self, chunk_ids: Iterable[str]) -> IndexingResult:
        ids = list(chunk_ids)
        if not ids:
            return IndexingResult()

        client = self._search_client()
        result = IndexingResult()
        batch_size = self._settings.upload_batch_size

        for start in range(0, len(ids), batch_size):
            window = ids[start : start + batch_size]
            try:
                responses = client.delete_documents(
                    documents=[{FIELD_CHUNK_ID: cid} for cid in window]
                )
            except Exception as exc:
                logger.error(
                    "delete batch failed outright",
                    extra={"index": self.index_name, "batch_size": len(window), "error": str(exc)},
                )
                result.failed.extend(
                    DocumentIndexError(chunk_id=cid, status_code=None, error_message=str(exc))
                    for cid in window
                )
                continue
            result += self._collect(responses, operation="delete")

        if result.failed:
            logger.error(
                "index delete completed with failures",
                extra={"index": self.index_name, **result.summary()},
            )
        return result

    def delete_document(self, document_id: str) -> IndexingResult:
        """Remove every chunk of one document — the re-index-from-scratch path."""
        ids = self.chunk_ids_for(document_id)
        logger.info(
            "deleting all chunks for document",
            extra={"index": self.index_name, "document_id": document_id, "chunks": len(ids)},
        )
        return self.delete(sorted(ids))

    def _collect(self, responses: Iterable[Any], *, operation: str) -> IndexingResult:
        """Split per-document responses into successes and failures, logging each
        rejection with its key and reason."""
        result = IndexingResult()
        for response in responses:
            key = getattr(response, "key", "") or ""
            if getattr(response, "succeeded", False):
                result.succeeded.append(key)
                continue
            status = getattr(response, "status_code", None)
            message = getattr(response, "error_message", None) or "unknown error"
            logger.error(
                "index operation rejected a document",
                extra={
                    "index": self.index_name,
                    "operation": operation,
                    "chunk_id": key,
                    "status_code": status,
                    "error": message,
                },
            )
            result.failed.append(
                DocumentIndexError(chunk_id=key, status_code=status, error_message=message)
            )
        return result

    # -- reads --------------------------------------------------------------

    def existing_hashes(self) -> dict[str, str]:
        """One content hash per indexed document.

        Filtered to ordinal 0, which every indexed document has, so this is one
        row per document rather than a full index scan.
        """
        from azure.core.exceptions import ResourceNotFoundError

        hashes: dict[str, str] = {}
        try:
            results = self._search_client().search(
                search_text="*",
                select=[FIELD_DOCUMENT_ID, FIELD_CONTENT_HASH],
                filter=f"{FIELD_ORDINAL} eq 0",
                top=_MAX_TOP,
            )
            for row in results:
                document_id = row.get(FIELD_DOCUMENT_ID)
                content_hash = row.get(FIELD_CONTENT_HASH)
                if document_id and content_hash:
                    hashes[document_id] = content_hash
        except ResourceNotFoundError:
            logger.info(
                "index does not exist yet, treating corpus as unindexed",
                extra={"index": self.index_name},
            )
        except Exception as exc:
            raise IndexingError(f"could not read existing content hashes: {exc}") from exc
        return hashes

    def chunk_ids_for(self, document_id: str) -> set[str]:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            results = self._search_client().search(
                search_text="*",
                select=[FIELD_CHUNK_ID],
                filter=f"{FIELD_DOCUMENT_ID} eq '{escape_odata(document_id)}'",
                top=_MAX_CHUNK_LISTING,
            )
            return {row[FIELD_CHUNK_ID] for row in results if row.get(FIELD_CHUNK_ID)}
        except ResourceNotFoundError:
            return set()
        except Exception as exc:
            raise IndexingError(f"could not list chunks for {document_id}: {exc}") from exc

    def search(
        self,
        *,
        query: str | None = None,
        vector: Sequence[float] | None = None,
        mode: SearchMode = SearchMode.HYBRID,
        top: int | None = None,
        filters: str | None = None,
        document_ids: Sequence[str] | None = None,
        document_types: Sequence[str] | None = None,
        use_semantic_ranker: bool | None = None,
    ) -> list[SearchHit]:
        """Keyword, vector or hybrid retrieval.

        Hybrid sends the text and the vector in one request; the service fuses
        the two result sets with Reciprocal Rank Fusion and, when the semantic
        ranker is enabled, re-scores the fused top candidates with a
        cross-encoder. That is one round trip, not two.
        """
        mode = SearchMode(mode)
        if mode in (SearchMode.KEYWORD, SearchMode.HYBRID) and not (query and query.strip()):
            raise ValueError(f"{mode} search requires a non-empty query")
        if mode in (SearchMode.VECTOR, SearchMode.HYBRID) and vector is None:
            raise ValueError(f"{mode} search requires a vector")
        if vector is not None and len(vector) != self.vector_dimensions:
            raise ValueError(
                f"vector has {len(vector)} dimensions, index expects {self.vector_dimensions}"
            )

        size = min(top or self._settings.default_top, _MAX_TOP)
        combined_filter = build_filter(
            filters=filters, document_ids=document_ids, document_types=document_types
        )
        semantic = (
            self._settings.use_semantic_ranker
            if use_semantic_ranker is None
            else use_semantic_ranker
        )
        # The semantic ranker re-scores lexical candidates; it contributes nothing
        # to a pure vector query.
        semantic = semantic and mode is not SearchMode.VECTOR

        kwargs: dict[str, Any] = {
            "select": list(RETRIEVAL_FIELDS),
            "top": size,
            "filter": combined_filter,
        }
        # A vector-only query must not let text drive BM25; "*" matches all so the
        # vector alone orders the results.
        kwargs["search_text"] = query if mode is not SearchMode.VECTOR else None

        if mode in (SearchMode.VECTOR, SearchMode.HYBRID):
            from azure.search.documents.models import VectorizedQuery

            kwargs["vector_queries"] = [
                VectorizedQuery(
                    vector=list(vector),
                    k_nearest_neighbors=size,
                    fields=FIELD_EMBEDDING,
                )
            ]

        if semantic:
            from azure.search.documents.models import QueryType

            kwargs["query_type"] = QueryType.SEMANTIC
            kwargs["semantic_configuration_name"] = SEMANTIC_CONFIG_NAME

        try:
            results = self._search_client().search(**kwargs)
            hits = [document_to_hit(row) for row in results]
        except Exception as exc:
            raise IndexingError(f"{mode} search failed: {exc}") from exc

        logger.debug(
            "search complete",
            extra={
                "index": self.index_name,
                "mode": str(mode),
                "semantic": semantic,
                "top": size,
                "hits": len(hits),
                "filter": combined_filter,
            },
        )
        return hits


def escape_odata(value: str) -> str:
    """Escape a string literal for an OData filter by doubling single quotes."""
    return value.replace("'", "''")


def build_filter(
    *,
    filters: str | None = None,
    document_ids: Sequence[str] | None = None,
    document_types: Sequence[str] | None = None,
) -> str | None:
    """Compose an OData filter from convenience arguments plus a raw clause."""
    clauses: list[str] = []
    if document_ids:
        clauses.append(_in_clause(FIELD_DOCUMENT_ID, document_ids))
    if document_types:
        clauses.append(_in_clause(FIELD_DOCUMENT_TYPE, document_types))
    if filters:
        clauses.append(f"({filters})")
    return " and ".join(clauses) if clauses else None


def _in_clause(field: str, values: Sequence[str]) -> str:
    """`search.in` stays a fixed-size expression however many values are passed,
    unlike a chain of `or`s which can breach the filter length limit.

    A pipe is used as the delimiter because document ids and types may contain
    commas but never pipes.
    """
    joined = "|".join(escape_odata(v) for v in values)
    return f"search.in({field}, '{joined}', '|')"
