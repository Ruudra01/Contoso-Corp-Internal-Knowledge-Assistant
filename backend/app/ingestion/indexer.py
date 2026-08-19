"""Indexing into Azure AI Search.

The index is the source of truth for what has already been ingested, which is
what makes the pipeline idempotent without a second datastore:

* `existing_hashes()` returns `{document_id: content_hash}` so an unchanged
  document is skipped before it is ever embedded;
* chunk keys are deterministic (`{document_id}-{ordinal:04d}`), so a re-index is
  a `mergeOrUpload` over the same keys rather than a duplicate insert;
* `prune()` deletes the keys a document no longer produces, so a document that
  shrinks does not leave orphaned chunks answering queries.

`InMemoryIndexer` implements the same contract for tests.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence, runtime_checkable

from app.core.config import AzureSearchSettings
from app.core.errors import ConfigurationError, IndexingError
from app.core.logging import get_logger
from app.ingestion.models import Chunk

logger = get_logger(__name__)

VECTOR_FIELD = "content_vector"
SEMANTIC_CONFIG_NAME = "contoso-semantic"
_VECTOR_PROFILE = "contoso-hnsw-profile"
_VECTOR_ALGORITHM = "contoso-hnsw"


@runtime_checkable
class Indexer(Protocol):
    def ensure_index(self) -> None: ...

    def existing_hashes(self) -> dict[str, str]: ...

    def chunk_ids_for(self, document_id: str) -> set[str]: ...

    def upsert(self, chunks: Sequence[Chunk]) -> int: ...

    def delete(self, chunk_ids: Iterable[str]) -> int: ...


def to_search_document(chunk: Chunk) -> dict:
    """Map a chunk onto the index schema. All required metadata travels here."""
    document = {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "document_type": str(chunk.document_type),
        "section": chunk.section,
        "section_path": chunk.section_path_text,
        "page_number": chunk.page_number,
        "page_end": chunk.page_end,
        "source_uri": chunk.source_uri,
        "ordinal": chunk.ordinal,
        "token_count": chunk.token_count,
        "content_hash": chunk.content_hash,
        "content": chunk.text,
    }
    if chunk.embedding is not None:
        document[VECTOR_FIELD] = chunk.embedding
    return document


class AzureSearchIndexer:
    """Azure AI Search implementation.

    Auth: an admin API key from configuration when present, otherwise
    `DefaultAzureCredential`. No credential is read from source.
    """

    def __init__(self, settings: AzureSearchSettings, *, vector_dimensions: int) -> None:
        if not settings.endpoint:
            raise ConfigurationError(
                "AZURE_SEARCH_ENDPOINT is not set; export it or use InMemoryIndexer for offline runs"
            )
        self._settings = settings
        self._vector_dimensions = vector_dimensions
        self._credential = self._build_credential(settings)

    @staticmethod
    def _build_credential(settings: AzureSearchSettings):
        if settings.api_key:
            from azure.core.credentials import AzureKeyCredential

            logger.info("search client authenticating with API key from configuration")
            return AzureKeyCredential(settings.api_key)

        from azure.identity import DefaultAzureCredential

        logger.info("search client authenticating with DefaultAzureCredential")
        return DefaultAzureCredential()

    def _search_client(self):
        from azure.search.documents import SearchClient

        return SearchClient(
            endpoint=self._settings.endpoint,
            index_name=self._settings.index_name,
            credential=self._credential,
        )

    def _index_client(self):
        from azure.search.documents.indexes import SearchIndexClient

        return SearchIndexClient(endpoint=self._settings.endpoint, credential=self._credential)

    # -- schema -------------------------------------------------------------

    def ensure_index(self) -> None:
        """Create the index if absent. Existing indexes are left untouched:
        changing a live schema is a migration, not an ingestion concern."""
        from azure.core.exceptions import ResourceNotFoundError

        client = self._index_client()
        try:
            client.get_index(self._settings.index_name)
            logger.info("search index present", extra={"index": self._settings.index_name})
            return
        except ResourceNotFoundError:
            pass

        logger.info("creating search index", extra={"index": self._settings.index_name})
        try:
            client.create_index(self._build_index())
        except Exception as exc:
            raise IndexingError(f"could not create index {self._settings.index_name}: {exc}") from exc

    def _build_index(self):
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SemanticConfiguration,
            SemanticField,
            SemanticPrioritizedFields,
            SemanticSearch,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        fields = [
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
            SimpleField(
                name="document_id", type=SearchFieldDataType.String, filterable=True, facetable=True
            ),
            SearchableField(name="document_name", type=SearchFieldDataType.String, filterable=True),
            SimpleField(
                name="document_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchableField(name="section", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="section_path", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="page_end", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="source_uri", type=SearchFieldDataType.String),
            SimpleField(name="ordinal", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="token_count", type=SearchFieldDataType.Int32),
            SimpleField(name="content_hash", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name=VECTOR_FIELD,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self._vector_dimensions,
                vector_search_profile_name=_VECTOR_PROFILE,
            ),
        ]
        return SearchIndex(
            name=self._settings.index_name,
            fields=fields,
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name=_VECTOR_ALGORITHM)],
                profiles=[
                    VectorSearchProfile(
                        name=_VECTOR_PROFILE, algorithm_configuration_name=_VECTOR_ALGORITHM
                    )
                ],
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name=SEMANTIC_CONFIG_NAME,
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="document_name"),
                            content_fields=[SemanticField(field_name="content")],
                            keywords_fields=[SemanticField(field_name="section_path")],
                        ),
                    )
                ]
            ),
        )

    # -- reads --------------------------------------------------------------

    def existing_hashes(self) -> dict[str, str]:
        """One content hash per already-indexed document."""
        from azure.core.exceptions import ResourceNotFoundError

        hashes: dict[str, str] = {}
        try:
            results = self._search_client().search(
                search_text="*",
                select=["document_id", "content_hash"],
                # Ordinal 0 always exists for an indexed document, so one row per
                # document is enough and avoids paging the whole index.
                filter="ordinal eq 0",
                top=1000,
            )
            for row in results:
                if (doc_id := row.get("document_id")) and (h := row.get("content_hash")):
                    hashes[doc_id] = h
        except ResourceNotFoundError:
            logger.info("index does not exist yet, treating corpus as unindexed")
        except Exception as exc:
            raise IndexingError(f"could not read existing content hashes: {exc}") from exc
        return hashes

    def chunk_ids_for(self, document_id: str) -> set[str]:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            results = self._search_client().search(
                search_text="*",
                select=["chunk_id"],
                filter=f"document_id eq '{_escape_odata(document_id)}'",
                top=10000,
            )
            return {row["chunk_id"] for row in results if row.get("chunk_id")}
        except ResourceNotFoundError:
            return set()
        except Exception as exc:
            raise IndexingError(f"could not list chunks for {document_id}: {exc}") from exc

    # -- writes -------------------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        client = self._search_client()
        written = 0
        batch_size = self._settings.upload_batch_size
        for start in range(0, len(chunks), batch_size):
            batch = [to_search_document(c) for c in chunks[start : start + batch_size]]
            try:
                results = client.merge_or_upload_documents(documents=batch)
            except Exception as exc:
                raise IndexingError(f"upsert failed: {exc}") from exc
            failures = [r for r in results if not r.succeeded]
            if failures:
                raise IndexingError(
                    f"{len(failures)}/{len(batch)} documents rejected, first: "
                    f"{failures[0].key} {failures[0].error_message}"
                )
            written += len(batch)
        return written

    def delete(self, chunk_ids: Iterable[str]) -> int:
        ids = list(chunk_ids)
        if not ids:
            return 0
        client = self._search_client()
        try:
            client.delete_documents(documents=[{"chunk_id": cid} for cid in ids])
        except Exception as exc:
            raise IndexingError(f"delete failed: {exc}") from exc
        return len(ids)


class InMemoryIndexer:
    """In-process index with the same semantics, for tests and dry runs."""

    def __init__(self) -> None:
        self.documents: dict[str, dict] = {}
        self.ensure_index_calls = 0
        self.deleted: list[str] = []

    def ensure_index(self) -> None:
        self.ensure_index_calls += 1

    def existing_hashes(self) -> dict[str, str]:
        return {
            doc["document_id"]: doc["content_hash"]
            for doc in self.documents.values()
            if doc.get("ordinal") == 0 and doc.get("content_hash")
        }

    def chunk_ids_for(self, document_id: str) -> set[str]:
        return {
            key for key, doc in self.documents.items() if doc["document_id"] == document_id
        }

    def upsert(self, chunks: Sequence[Chunk]) -> int:
        for chunk in chunks:
            self.documents[chunk.chunk_id] = to_search_document(chunk)
        return len(chunks)

    def delete(self, chunk_ids: Iterable[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if self.documents.pop(chunk_id, None) is not None:
                self.deleted.append(chunk_id)
                removed += 1
        return removed


def _escape_odata(value: str) -> str:
    """Single quotes are escaped by doubling them in OData filters."""
    return value.replace("'", "''")


def build_indexer(
    settings: AzureSearchSettings, *, vector_dimensions: int, offline: bool = False
) -> Indexer:
    if offline or not settings.endpoint:
        logger.warning(
            "using InMemoryIndexer: nothing is persisted",
            extra={"reason": "offline flag" if offline else "AZURE_SEARCH_ENDPOINT unset"},
        )
        return InMemoryIndexer()
    return AzureSearchIndexer(settings, vector_dimensions=vector_dimensions)
