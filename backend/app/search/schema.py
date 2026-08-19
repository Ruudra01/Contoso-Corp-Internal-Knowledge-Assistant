"""Index schema: field names, their search behaviour, and the mapping to and
from chunk objects.

Field configuration rationale (requirement: searchable/filterable set correctly).

Searchable (fed to BM25 and the semantic ranker):
    chunk_text      the evidence itself, the primary lexical target
    document_name   users name documents ("the travel policy")
    section         users name sections ("carryover")
    section_path    carries the breadcrumb terms for keyword recall

Filterable (used for scoping, dedupe and idempotency, never scored):
    document_id     per-document scoping and stale-chunk pruning
    document_type   filter by format
    content_hash    the skip-unchanged short circuit
    ordinal         one-row-per-document lookups and ordering
    page_number     page-range filters
    section, section_path, document_name are filterable too, so retrieval can
    scope to a section without a second round trip.

Retrieve-only (returned, never searched or filtered):
    source_uri      a URL contributes nothing to BM25 and would pollute scoring
    page_end, token_count

`chunk_text` is intentionally NOT filterable: filtering on a large free-text
field is a service-side error waiting to happen and BM25 already covers it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.search.models import SearchHit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.ingestion.models import Chunk

# -- field names ---------------------------------------------------------------
# Constants rather than string literals so a rename is a single edit and no
# module outside this one hard-codes the wire format.
FIELD_CHUNK_ID = "chunk_id"
FIELD_DOCUMENT_ID = "document_id"
FIELD_DOCUMENT_NAME = "document_name"
FIELD_DOCUMENT_TYPE = "document_type"
FIELD_SECTION = "section"
FIELD_SECTION_PATH = "section_path"
FIELD_CHUNK_TEXT = "chunk_text"
FIELD_PAGE_NUMBER = "page_number"
FIELD_PAGE_END = "page_end"
FIELD_SOURCE_URI = "source_uri"
FIELD_ORDINAL = "ordinal"
FIELD_TOKEN_COUNT = "token_count"
FIELD_CONTENT_HASH = "content_hash"
FIELD_EMBEDDING = "embedding"

KEY_FIELD = FIELD_CHUNK_ID

# Fields worth returning on a query. The embedding is excluded on purpose: it is
# thousands of floats per hit and no caller needs it back.
RETRIEVAL_FIELDS = (
    FIELD_CHUNK_ID,
    FIELD_DOCUMENT_ID,
    FIELD_DOCUMENT_NAME,
    FIELD_DOCUMENT_TYPE,
    FIELD_SECTION,
    FIELD_SECTION_PATH,
    FIELD_CHUNK_TEXT,
    FIELD_PAGE_NUMBER,
    FIELD_PAGE_END,
    FIELD_SOURCE_URI,
    FIELD_ORDINAL,
)

# Named configurations referenced by queries.
SEMANTIC_CONFIG_NAME = "contoso-semantic"
VECTOR_PROFILE_NAME = "contoso-hnsw-profile"
VECTOR_ALGORITHM_NAME = "contoso-hnsw"
# Microsoft's English analyzer: lemmatisation and stopword handling measurably
# beat the default on policy prose ("carrying over" -> "carry over").
TEXT_ANALYZER = "en.microsoft"

_SCORE = "@search.score"
_RERANKER_SCORE = "@search.reranker_score"


def chunk_to_document(chunk: "Chunk") -> dict[str, Any]:
    """Map a `Chunk` onto an index document.

    The embedding is omitted when absent so a dry run or a metadata-only merge
    does not blank out a stored vector.
    """
    document: dict[str, Any] = {
        FIELD_CHUNK_ID: chunk.chunk_id,
        FIELD_DOCUMENT_ID: chunk.document_id,
        FIELD_DOCUMENT_NAME: chunk.document_name,
        FIELD_DOCUMENT_TYPE: str(chunk.document_type),
        FIELD_SECTION: chunk.section,
        FIELD_SECTION_PATH: chunk.section_path_text,
        FIELD_CHUNK_TEXT: chunk.text,
        FIELD_PAGE_NUMBER: chunk.page_number,
        FIELD_PAGE_END: chunk.page_end,
        FIELD_SOURCE_URI: chunk.source_uri,
        FIELD_ORDINAL: chunk.ordinal,
        FIELD_TOKEN_COUNT: chunk.token_count,
        FIELD_CONTENT_HASH: chunk.content_hash,
    }
    if chunk.embedding is not None:
        document[FIELD_EMBEDDING] = chunk.embedding
    return document


def document_to_hit(document: dict[str, Any]) -> SearchHit:
    """Map an index document back to a `SearchHit`.

    Tolerant of missing fields: a `select` narrower than the full schema, or a
    document written by an older ingestion run, must not raise here.
    """
    return SearchHit(
        chunk_id=document.get(FIELD_CHUNK_ID, ""),
        document_id=document.get(FIELD_DOCUMENT_ID, ""),
        document_name=document.get(FIELD_DOCUMENT_NAME, ""),
        document_type=document.get(FIELD_DOCUMENT_TYPE, ""),
        section=document.get(FIELD_SECTION, ""),
        section_path=document.get(FIELD_SECTION_PATH, ""),
        chunk_text=document.get(FIELD_CHUNK_TEXT, ""),
        source_uri=document.get(FIELD_SOURCE_URI, ""),
        score=float(document.get(_SCORE) or 0.0),
        page_number=document.get(FIELD_PAGE_NUMBER),
        page_end=document.get(FIELD_PAGE_END),
        ordinal=document.get(FIELD_ORDINAL),
        reranker_score=(
            float(document[_RERANKER_SCORE]) if document.get(_RERANKER_SCORE) is not None else None
        ),
    )


def build_index(index_name: str, vector_dimensions: int):
    """Construct the Azure AI Search index definition.

    Imports the SDK lazily so that modules which only need the field names or
    the mappers above do not pull in `azure-search-documents`.
    """
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        HnswParameters,
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
        VectorSearchAlgorithmMetric,
        VectorSearchProfile,
    )

    fields = [
        SimpleField(name=FIELD_CHUNK_ID, type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(
            name=FIELD_DOCUMENT_ID,
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            sortable=True,
        ),
        SearchableField(
            name=FIELD_DOCUMENT_NAME,
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            analyzer_name=TEXT_ANALYZER,
        ),
        SimpleField(
            name=FIELD_DOCUMENT_TYPE,
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchableField(
            name=FIELD_SECTION,
            type=SearchFieldDataType.String,
            filterable=True,
            analyzer_name=TEXT_ANALYZER,
        ),
        SearchableField(
            name=FIELD_SECTION_PATH,
            type=SearchFieldDataType.String,
            filterable=True,
            analyzer_name=TEXT_ANALYZER,
        ),
        SearchableField(
            name=FIELD_CHUNK_TEXT,
            type=SearchFieldDataType.String,
            analyzer_name=TEXT_ANALYZER,
        ),
        SimpleField(
            name=FIELD_PAGE_NUMBER,
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(name=FIELD_PAGE_END, type=SearchFieldDataType.Int32, filterable=True),
        SimpleField(name=FIELD_SOURCE_URI, type=SearchFieldDataType.String),
        SimpleField(
            name=FIELD_ORDINAL, type=SearchFieldDataType.Int32, filterable=True, sortable=True
        ),
        SimpleField(name=FIELD_TOKEN_COUNT, type=SearchFieldDataType.Int32),
        SimpleField(name=FIELD_CONTENT_HASH, type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name=FIELD_EMBEDDING,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            # A vector field is retrievable by default; suppressing it keeps
            # thousands of floats per hit off the wire.
            hidden=True,
            vector_search_dimensions=vector_dimensions,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=VECTOR_ALGORITHM_NAME,
                # Cosine matches how `text-embedding-3-*` vectors are compared.
                parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=VECTOR_ALGORITHM_NAME,
            )
        ],
    )

    semantic_search = SemanticSearch(
        default_configuration_name=SEMANTIC_CONFIG_NAME,
        configurations=[
            SemanticConfiguration(
                name=SEMANTIC_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name=FIELD_DOCUMENT_NAME),
                    content_fields=[SemanticField(field_name=FIELD_CHUNK_TEXT)],
                    keywords_fields=[SemanticField(field_name=FIELD_SECTION_PATH)],
                ),
            )
        ],
    )

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def vector_dimensions_of(index) -> int | None:
    """Read the configured vector width off an existing index definition."""
    for field in index.fields:
        if field.name == FIELD_EMBEDDING:
            return getattr(field, "vector_search_dimensions", None)
    return None
