"""Search layer: the application's only route to the chunk index.

Import from here rather than from `azure_store`, so callers stay independent of
the backing engine.
"""

from __future__ import annotations

from app.core.config import AzureSearchSettings
from app.core.logging import get_logger
from app.search.azure_store import AzureSearchStore, build_filter, escape_odata
from app.search.memory_store import InMemorySearchStore
from app.search.models import (
    DocumentIndexError,
    IndexHealth,
    IndexingResult,
    SearchHit,
    SearchMode,
)
from app.search.schema import (
    FIELD_CHUNK_ID,
    FIELD_CHUNK_TEXT,
    FIELD_DOCUMENT_ID,
    FIELD_DOCUMENT_NAME,
    FIELD_DOCUMENT_TYPE,
    FIELD_EMBEDDING,
    FIELD_PAGE_NUMBER,
    FIELD_SECTION,
    FIELD_SECTION_PATH,
    FIELD_SOURCE_URI,
    RETRIEVAL_FIELDS,
    SEMANTIC_CONFIG_NAME,
    build_index,
    chunk_to_document,
    document_to_hit,
)
from app.search.store import SearchStore

logger = get_logger(__name__)

__all__ = [
    "AzureSearchStore",
    "DocumentIndexError",
    "FIELD_CHUNK_ID",
    "FIELD_CHUNK_TEXT",
    "FIELD_DOCUMENT_ID",
    "FIELD_DOCUMENT_NAME",
    "FIELD_DOCUMENT_TYPE",
    "FIELD_EMBEDDING",
    "FIELD_PAGE_NUMBER",
    "FIELD_SECTION",
    "FIELD_SECTION_PATH",
    "FIELD_SOURCE_URI",
    "InMemorySearchStore",
    "IndexHealth",
    "IndexingResult",
    "RETRIEVAL_FIELDS",
    "SEMANTIC_CONFIG_NAME",
    "SearchHit",
    "SearchMode",
    "SearchStore",
    "build_filter",
    "build_index",
    "build_search_store",
    "chunk_to_document",
    "document_to_hit",
    "escape_odata",
]


def build_search_store(
    settings: AzureSearchSettings, *, vector_dimensions: int, offline: bool = False
) -> SearchStore:
    """Pick a backend. `offline`, or a missing endpoint, selects the local one."""
    if offline or not settings.endpoint:
        logger.warning(
            "using InMemorySearchStore: nothing is persisted and ranking is not semantic",
            extra={"reason": "offline flag" if offline else "AZURE_SEARCH_ENDPOINT unset"},
        )
        return InMemorySearchStore(
            vector_dimensions=vector_dimensions, index_name=settings.index_name
        )
    return AzureSearchStore(settings, vector_dimensions=vector_dimensions)
