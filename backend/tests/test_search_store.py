"""Search layer: schema configuration, indexing, deletion, health and the three
retrieval modes.

The Azure client is replaced by fakes that record calls and can be told to fail,
so these tests assert this code's own logic — field configuration, batching,
partial-failure handling, filter composition, query construction — without a live
service. `InMemorySearchStore` is exercised directly for ranking behaviour.
"""

from __future__ import annotations

import pytest

from app.core.config import AzureSearchSettings
from app.core.errors import ConfigurationError, IndexingError
from app.ingestion.models import Chunk, DocumentType
from app.search import (
    AzureSearchStore,
    InMemorySearchStore,
    IndexingResult,
    SearchHit,
    SearchMode,
    SearchStore,
    build_filter,
    build_index,
    build_search_store,
    chunk_to_document,
    document_to_hit,
    escape_odata,
)
from app.search.schema import (
    FIELD_CHUNK_TEXT,
    FIELD_EMBEDDING,
    SEMANTIC_CONFIG_NAME,
    vector_dimensions_of,
)

DIMENSIONS = 8
NOT_A_REAL_KEY = "unit-test-placeholder-not-a-credential"


def chunk(
    ordinal: int = 0,
    *,
    document_id: str = "CNT-HR-005",
    text: str = "Employees may carry over five unused PTO days.",
    document_type: DocumentType = DocumentType.PDF,
    embedding: list[float] | None = None,
) -> Chunk:
    made = Chunk(
        document_id=document_id,
        document_name="Paid Time Off Policy",
        document_type=document_type,
        source_uri="https://acct.blob.core.windows.net/corpus/raw/pto.pdf",
        ordinal=ordinal,
        text=text,
        embedding_text=f"Paid Time Off Policy > Carryover\n\n{text}",
        section="Carryover",
        section_path=("Paid Time Off Policy", "Carryover"),
        token_count=12,
        page_number=2,
        page_end=2,
        content_hash="hash-1",
    )
    made.embedding = embedding if embedding is not None else [0.0] * DIMENSIONS
    return made


def azure_store(dimensions: int = DIMENSIONS) -> AzureSearchStore:
    return AzureSearchStore(
        AzureSearchSettings(
            endpoint="https://svc.search.windows.net",
            api_key=NOT_A_REAL_KEY,
            index_name="contoso-policies-test",
        ),
        vector_dimensions=dimensions,
    )


# =============================================================================
# Schema: searchable / filterable / vector configuration
# =============================================================================


def test_index_contains_every_required_field() -> None:
    fields = {f.name for f in build_index("i", 3072).fields}

    required = {
        "chunk_id",
        "document_id",
        "document_name",
        "document_type",
        "section",
        "section_path",
        "chunk_text",
        "page_number",
        "source_uri",
        "embedding",
    }
    assert required <= fields


def test_searchable_fields_are_the_lexical_targets() -> None:
    fields = {f.name: f for f in build_index("i", 3072).fields}

    for name in ("chunk_text", "document_name", "section", "section_path"):
        assert fields[name].searchable, f"{name} must be searchable for keyword search"
    # A URL and an opaque key contribute nothing to BM25 and would pollute scoring.
    for name in ("source_uri", "chunk_id", "document_type", "content_hash"):
        assert not fields[name].searchable, f"{name} must not be searchable"


def test_filterable_fields_cover_scoping_and_idempotency() -> None:
    fields = {f.name: f for f in build_index("i", 3072).fields}

    for name in ("document_id", "document_type", "content_hash", "ordinal", "page_number"):
        assert fields[name].filterable, f"{name} must be filterable"
    # Filtering on a large free-text field is a service-side error waiting to happen.
    assert not fields[FIELD_CHUNK_TEXT].filterable


def test_key_field_is_the_chunk_id() -> None:
    fields = {f.name: f for f in build_index("i", 3072).fields}

    assert fields["chunk_id"].key is True
    assert sum(bool(getattr(f, "key", False)) for f in build_index("i", 3072).fields) == 1


def test_vector_field_matches_the_embedding_model() -> None:
    index = build_index("i", 3072)
    fields = {f.name: f for f in index.fields}
    vector = fields[FIELD_EMBEDDING]

    # text-embedding-3-large is 3072-dimensional.
    assert vector.vector_search_dimensions == 3072
    assert vector.searchable
    assert vector.vector_search_profile_name == index.vector_search.profiles[0].name
    # Returning thousands of floats per hit would be pure waste.
    assert vector.hidden is True


def test_vector_dimensions_follow_configuration() -> None:
    assert vector_dimensions_of(build_index("i", 1536)) == 1536
    assert vector_dimensions_of(build_index("i", 3072)) == 3072


def test_vector_search_uses_cosine_similarity() -> None:
    algorithm = build_index("i", 3072).vector_search.algorithms[0]

    assert str(algorithm.parameters.metric).lower().endswith("cosine")


def test_semantic_configuration_is_wired_for_reranking() -> None:
    semantic = build_index("i", 3072).semantic_search
    configuration = semantic.configurations[0]

    assert configuration.name == SEMANTIC_CONFIG_NAME
    assert semantic.default_configuration_name == SEMANTIC_CONFIG_NAME
    assert configuration.prioritized_fields.title_field.field_name == "document_name"
    assert configuration.prioritized_fields.content_fields[0].field_name == FIELD_CHUNK_TEXT


def test_index_name_is_configurable() -> None:
    assert build_index("contoso-policies-v2", 3072).name == "contoso-policies-v2"
    store = AzureSearchStore(
        AzureSearchSettings(
            endpoint="https://svc.search.windows.net", api_key=NOT_A_REAL_KEY, index_name="custom"
        ),
        vector_dimensions=DIMENSIONS,
    )
    assert store.index_name == "custom"


def test_text_fields_use_an_english_analyzer() -> None:
    """Lemmatisation matters on policy prose: "carrying over" must match "carry over"."""
    fields = {f.name: f for f in build_index("i", 3072).fields}

    assert "en.microsoft" in str(fields[FIELD_CHUNK_TEXT].analyzer_name).lower().replace("_", ".")


# =============================================================================
# Document mapping
# =============================================================================


def test_chunk_maps_onto_every_required_field() -> None:
    document = chunk_to_document(chunk())

    assert document["chunk_id"] == "CNT-HR-005-0000"
    assert document["document_id"] == "CNT-HR-005"
    assert document["document_name"] == "Paid Time Off Policy"
    assert document["document_type"] == "pdf"
    assert document["section"] == "Carryover"
    assert document["section_path"] == "Paid Time Off Policy > Carryover"
    assert document["chunk_text"].startswith("Employees may carry over")
    assert document["page_number"] == 2
    assert document["source_uri"].endswith("pto.pdf")
    assert len(document["embedding"]) == DIMENSIONS


def test_embedding_is_omitted_when_absent() -> None:
    """A metadata-only merge must not blank out a stored vector."""
    bare = chunk()
    bare.embedding = None

    assert FIELD_EMBEDDING not in chunk_to_document(bare)


def test_document_maps_back_to_a_hit() -> None:
    document = chunk_to_document(chunk())
    document["@search.score"] = 0.53
    document["@search.reranker_score"] = 2.41

    hit = document_to_hit(document)

    assert isinstance(hit, SearchHit)
    assert hit.chunk_id == "CNT-HR-005-0000"
    assert hit.score == pytest.approx(0.53)
    assert hit.reranker_score == pytest.approx(2.41)
    # The reranker score wins when present; the scales are not comparable.
    assert hit.ranking_score == pytest.approx(2.41)
    assert hit.citation == "Paid Time Off Policy - Paid Time Off Policy > Carryover"


def test_hit_mapping_tolerates_a_narrow_select() -> None:
    hit = document_to_hit({"chunk_id": "c-1"})

    assert hit.chunk_id == "c-1"
    assert hit.score == 0.0
    assert hit.reranker_score is None


# =============================================================================
# Configuration and secrets
# =============================================================================


def test_endpoint_is_required() -> None:
    with pytest.raises(ConfigurationError, match="AZURE_SEARCH_ENDPOINT"):
        AzureSearchStore(AzureSearchSettings(endpoint=None), vector_dimensions=DIMENSIONS)


def test_vector_dimensions_must_be_positive() -> None:
    with pytest.raises(ConfigurationError, match="vector_dimensions"):
        AzureSearchStore(
            AzureSearchSettings(endpoint="https://svc.search.windows.net", api_key=NOT_A_REAL_KEY),
            vector_dimensions=0,
        )


def test_repr_does_not_leak_the_api_key() -> None:
    text = repr(azure_store())

    assert NOT_A_REAL_KEY not in text
    assert "contoso-policies-test" in text


def test_factory_falls_back_offline_without_an_endpoint() -> None:
    store = build_search_store(AzureSearchSettings(endpoint=None), vector_dimensions=DIMENSIONS)

    assert isinstance(store, InMemorySearchStore)


def test_factory_honours_the_offline_flag() -> None:
    store = build_search_store(
        AzureSearchSettings(endpoint="https://svc.search.windows.net", api_key=NOT_A_REAL_KEY),
        vector_dimensions=DIMENSIONS,
        offline=True,
    )

    assert isinstance(store, InMemorySearchStore)


def test_both_backends_satisfy_the_abstraction() -> None:
    assert isinstance(InMemorySearchStore(), SearchStore)
    assert isinstance(azure_store(), SearchStore)


# =============================================================================
# Index creation and update
# =============================================================================


class FakeIndexClient:
    """Stands in for `SearchIndexClient`."""

    def __init__(self, *, existing=None, statistics=None) -> None:
        self.existing = existing
        self.statistics = statistics if statistics is not None else {"document_count": 7}
        self.created: list = []
        self.updated: list = []
        self.deleted: list[str] = []

    def get_index(self, name):
        from azure.core.exceptions import ResourceNotFoundError

        if self.existing is None:
            raise ResourceNotFoundError("no such index")
        return self.existing

    def create_index(self, definition):
        self.created.append(definition)
        return definition

    def create_or_update_index(self, definition):
        self.updated.append(definition)
        return definition

    def get_index_statistics(self, name):
        return self.statistics

    def delete_index(self, name):
        self.deleted.append(name)


def with_index_client(store: AzureSearchStore, fake: FakeIndexClient) -> AzureSearchStore:
    store._index_client = lambda: fake
    return store


def test_ensure_index_creates_when_absent() -> None:
    fake = FakeIndexClient(existing=None)
    store = with_index_client(azure_store(), fake)

    created = store.ensure_index()

    assert created is True
    assert fake.created and fake.created[0].name == "contoso-policies-test"
    assert not fake.updated


def test_ensure_index_leaves_an_existing_schema_alone_by_default() -> None:
    fake = FakeIndexClient(existing=build_index("contoso-policies-test", DIMENSIONS))
    store = with_index_client(azure_store(), fake)

    created = store.ensure_index()

    assert created is False
    assert not fake.created and not fake.updated


def test_ensure_index_updates_when_explicitly_allowed() -> None:
    fake = FakeIndexClient(existing=build_index("contoso-policies-test", DIMENSIONS))
    store = with_index_client(azure_store(), fake)

    updated = store.ensure_index(allow_update=True)

    assert updated is True
    assert fake.updated and not fake.created


def test_ensure_index_refuses_a_vector_width_change() -> None:
    """Azure cannot re-dimension a populated vector field, so a silent no-op here
    would leave the index permanently inconsistent with the embedding model."""
    fake = FakeIndexClient(existing=build_index("contoso-policies-test", 1536))
    store = with_index_client(azure_store(dimensions=3072), fake)

    with pytest.raises(IndexingError, match="cannot re-dimension"):
        store.ensure_index(allow_update=True)


def test_index_creation_failure_is_wrapped() -> None:
    fake = FakeIndexClient(existing=None)
    fake.create_index = lambda definition: (_ for _ in ()).throw(RuntimeError("403 forbidden"))
    store = with_index_client(azure_store(), fake)

    with pytest.raises(IndexingError, match="could not create index"):
        store.ensure_index()


def test_delete_index_reports_failure_clearly() -> None:
    fake = FakeIndexClient(existing=build_index("contoso-policies-test", DIMENSIONS))
    fake.delete_index = lambda name: (_ for _ in ()).throw(RuntimeError("409 conflict"))
    store = with_index_client(azure_store(), fake)

    with pytest.raises(IndexingError, match="could not delete index"):
        store.delete_index()


# =============================================================================
# Health check
# =============================================================================


def test_health_reports_ready_when_the_index_exists() -> None:
    fake = FakeIndexClient(existing=build_index("contoso-policies-test", DIMENSIONS))
    store = with_index_client(azure_store(), fake)

    health = store.health()

    assert health.reachable and health.index_exists and health.ready
    assert health.document_count == 7
    assert health.vector_dimensions == DIMENSIONS
    assert health.error is None


def test_health_distinguishes_a_missing_index_from_an_unreachable_service() -> None:
    store = with_index_client(azure_store(), FakeIndexClient(existing=None))

    health = store.health()

    # The service answered, so it is reachable; the index simply is not there.
    assert health.reachable is True
    assert health.index_exists is False
    assert health.ready is False
    assert "does not exist" in health.error


def test_health_reports_an_unreachable_service_without_raising() -> None:
    fake = FakeIndexClient(existing=None)
    fake.get_index = lambda name: (_ for _ in ()).throw(RuntimeError("DNS failure"))
    store = with_index_client(azure_store(), fake)

    health = store.health()

    assert health.reachable is False
    assert health.ready is False
    assert "DNS failure" in health.error


def test_health_survives_missing_statistics() -> None:
    fake = FakeIndexClient(existing=build_index("contoso-policies-test", DIMENSIONS))
    fake.get_index_statistics = lambda name: (_ for _ in ()).throw(RuntimeError("no permission"))
    store = with_index_client(azure_store(), fake)

    health = store.health()

    # Statistics are informational; their absence is not unhealthy.
    assert health.ready is True
    assert health.document_count is None


def test_in_memory_health_reflects_stored_documents() -> None:
    store = InMemorySearchStore(vector_dimensions=DIMENSIONS)
    store.upsert([chunk(0), chunk(1)])

    health = store.health()

    assert health.ready and health.document_count == 2


# =============================================================================
# Batch indexing and partial failures
# =============================================================================


class FakeSearchClient:
    """Records upload batches and can reject chosen keys or whole batches."""

    def __init__(self, *, reject: set[str] | None = None, raise_on_batch: int | None = None) -> None:
        self.reject = reject or set()
        self.raise_on_batch = raise_on_batch
        self.batches: list[list[dict]] = []
        self.deleted_batches: list[list[dict]] = []
        self.search_calls: list[dict] = []
        self.rows: list[dict] = []

    def _respond(self, documents, key_field="chunk_id"):
        results = []
        for document in documents:
            key = document[key_field]
            rejected = key in self.reject
            results.append(
                type(
                    "R",
                    (),
                    {
                        "key": key,
                        "succeeded": not rejected,
                        "status_code": 400 if rejected else 200,
                        "error_message": "field too large" if rejected else None,
                    },
                )()
            )
        return results

    def merge_or_upload_documents(self, *, documents):
        index = len(self.batches)
        self.batches.append(list(documents))
        if self.raise_on_batch == index:
            raise RuntimeError("503 service unavailable")
        return self._respond(documents)

    def delete_documents(self, *, documents):
        self.deleted_batches.append(list(documents))
        return self._respond(documents)

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return iter(self.rows)


def with_search_client(store: AzureSearchStore, fake: FakeSearchClient) -> AzureSearchStore:
    store._search_client = lambda: fake
    return store


def test_upsert_batches_by_configured_size() -> None:
    settings = AzureSearchSettings(
        endpoint="https://svc.search.windows.net", api_key=NOT_A_REAL_KEY, upload_batch_size=2
    )
    store = AzureSearchStore(settings, vector_dimensions=DIMENSIONS)
    fake = FakeSearchClient()
    with_search_client(store, fake)

    result = store.upsert([chunk(i) for i in range(5)])

    assert [len(b) for b in fake.batches] == [2, 2, 1]
    assert result.succeeded_count == 5
    assert result.ok


def test_upsert_reports_partial_failure_without_discarding_successes() -> None:
    """A batch Azure accepts partially must report both sides, not a bare boolean."""
    store = with_search_client(azure_store(), FakeSearchClient(reject={"CNT-HR-005-0002"}))

    result = store.upsert([chunk(i) for i in range(4)])

    assert result.succeeded_count == 3
    assert result.failed_count == 1
    assert result.ok is False
    failure = result.failed[0]
    assert failure.chunk_id == "CNT-HR-005-0002"
    assert failure.status_code == 400
    assert failure.error_message == "field too large"


def test_a_whole_batch_failure_is_attributed_to_every_key() -> None:
    """A transport error must not leave keys silently reported as written."""
    settings = AzureSearchSettings(
        endpoint="https://svc.search.windows.net", api_key=NOT_A_REAL_KEY, upload_batch_size=2
    )
    store = AzureSearchStore(settings, vector_dimensions=DIMENSIONS)
    with_search_client(store, FakeSearchClient(raise_on_batch=0))

    result = store.upsert([chunk(i) for i in range(4)])

    assert result.failed_count == 2
    assert result.succeeded_count == 2
    assert all("503" in f.error_message for f in result.failed)


def test_upsert_is_idempotent_on_deterministic_keys() -> None:
    store = InMemorySearchStore(vector_dimensions=DIMENSIONS)
    chunks = [chunk(0), chunk(1)]

    store.upsert(chunks)
    store.upsert(chunks)

    assert len(store.documents) == 2


def test_empty_writes_do_not_call_the_service() -> None:
    store = azure_store()
    store._search_client = lambda: pytest.fail("must not call the service")

    assert store.upsert([]).succeeded_count == 0
    assert store.delete([]).succeeded_count == 0


def test_delete_removes_named_keys() -> None:
    store = with_search_client(azure_store(), FakeSearchClient())

    result = store.delete(["CNT-HR-005-0000", "CNT-HR-005-0001"])

    assert result.succeeded_count == 2
    assert result.ok


def test_delete_reports_rejected_keys() -> None:
    store = with_search_client(azure_store(), FakeSearchClient(reject={"CNT-HR-005-0001"}))

    result = store.delete(["CNT-HR-005-0000", "CNT-HR-005-0001"])

    assert result.succeeded_count == 1 and result.failed_count == 1


def test_delete_document_removes_every_chunk_of_that_document() -> None:
    store = InMemorySearchStore(vector_dimensions=DIMENSIONS)
    store.upsert([chunk(0), chunk(1), chunk(0, document_id="CNT-HR-019")])

    result = store.delete_document("CNT-HR-005")

    assert result.succeeded_count == 2
    assert set(store.documents) == {"CNT-HR-019-0000"}


def test_reindex_replaces_content_without_duplicating_keys() -> None:
    store = InMemorySearchStore(vector_dimensions=DIMENSIONS)
    store.upsert([chunk(0, text="Original text.")])

    store.upsert([chunk(0, text="Updated text.")])

    assert len(store.documents) == 1
    assert store.documents["CNT-HR-005-0000"]["chunk_text"] == "Updated text."


def test_indexing_result_addition_accumulates_both_sides() -> None:
    from app.search.models import DocumentIndexError

    combined = IndexingResult(succeeded=["a"]) + IndexingResult(
        failed=[DocumentIndexError("b", 400, "bad")]
    )

    assert combined.succeeded == ["a"]
    assert combined.failed_count == 1
    assert combined.summary()["first_error"] == "bad"


# =============================================================================
# Idempotency reads
# =============================================================================


def test_existing_hashes_returns_one_row_per_document() -> None:
    store = InMemorySearchStore(vector_dimensions=DIMENSIONS)
    store.upsert([chunk(0), chunk(1), chunk(0, document_id="CNT-HR-019")])

    assert store.existing_hashes() == {"CNT-HR-005": "hash-1", "CNT-HR-019": "hash-1"}


def test_existing_hashes_filters_to_ordinal_zero() -> None:
    """Reading one row per document beats paging the whole index."""
    store = with_search_client(azure_store(), FakeSearchClient())

    store.existing_hashes()

    call = store._search_client().search_calls[-1]
    assert call["filter"] == "ordinal eq 0"


def test_existing_hashes_treats_a_missing_index_as_unindexed() -> None:
    from azure.core.exceptions import ResourceNotFoundError

    fake = FakeSearchClient()
    fake.search = lambda **kwargs: (_ for _ in ()).throw(ResourceNotFoundError("no index"))
    store = with_search_client(azure_store(), fake)

    assert store.existing_hashes() == {}


def test_existing_hashes_raises_on_a_real_failure() -> None:
    """Silently returning {} would trigger a full re-embed of the corpus."""
    fake = FakeSearchClient()
    fake.search = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("401 unauthorized"))
    store = with_search_client(azure_store(), fake)

    with pytest.raises(IndexingError, match="could not read existing content hashes"):
        store.existing_hashes()


def test_chunk_ids_for_scopes_by_document() -> None:
    store = InMemorySearchStore(vector_dimensions=DIMENSIONS)
    store.upsert([chunk(0), chunk(1), chunk(0, document_id="CNT-HR-019")])

    assert store.chunk_ids_for("CNT-HR-005") == {"CNT-HR-005-0000", "CNT-HR-005-0001"}


# =============================================================================
# Filters
# =============================================================================


def test_filter_is_none_when_nothing_is_requested() -> None:
    assert build_filter() is None


def test_document_and_type_filters_compose() -> None:
    built = build_filter(document_ids=["CNT-HR-005"], document_types=["pdf", "docx"])

    assert "document_id" in built and "document_type" in built
    assert " and " in built


def test_raw_filter_is_combined_with_the_convenience_arguments() -> None:
    built = build_filter(document_types=["pdf"], filters="page_number eq 2")

    assert built.endswith("(page_number eq 2)")


def test_odata_string_literals_are_escaped() -> None:
    assert escape_odata("O'Brien") == "O''Brien"
    assert "O''Brien" in build_filter(document_ids=["O'Brien"])


# =============================================================================
# Keyword, vector and hybrid search
# =============================================================================


def vector_of(value: float) -> list[float]:
    return [value] * DIMENSIONS


def test_keyword_search_sends_text_and_no_vector() -> None:
    store = with_search_client(azure_store(), FakeSearchClient())

    store.search(query="carry over unused PTO", mode=SearchMode.KEYWORD, top=5)

    call = store._search_client().search_calls[-1]
    assert call["search_text"] == "carry over unused PTO"
    assert "vector_queries" not in call
    assert call["top"] == 5


def test_vector_search_sends_a_vector_and_no_text() -> None:
    store = with_search_client(azure_store(), FakeSearchClient())

    store.search(vector=vector_of(0.2), mode=SearchMode.VECTOR, top=3)

    call = store._search_client().search_calls[-1]
    assert call["search_text"] is None
    assert len(call["vector_queries"]) == 1
    assert call["vector_queries"][0].fields == FIELD_EMBEDDING
    # The semantic ranker re-scores lexical candidates; a pure vector query has none.
    assert "query_type" not in call


def test_hybrid_search_sends_text_and_vector_in_one_request() -> None:
    store = with_search_client(azure_store(), FakeSearchClient())

    store.search(query="unused PTO", vector=vector_of(0.2), mode=SearchMode.HYBRID)

    calls = store._search_client().search_calls
    assert len(calls) == 1, "hybrid must be one round trip, not two"
    assert calls[0]["search_text"] == "unused PTO"
    assert calls[0]["vector_queries"]


def test_hybrid_enables_the_semantic_ranker_by_default() -> None:
    store = with_search_client(azure_store(), FakeSearchClient())

    store.search(query="unused PTO", vector=vector_of(0.2))

    call = store._search_client().search_calls[-1]
    assert call["semantic_configuration_name"] == SEMANTIC_CONFIG_NAME
    assert str(call["query_type"]).lower().endswith("semantic")


def test_semantic_ranker_can_be_disabled_per_query() -> None:
    store = with_search_client(azure_store(), FakeSearchClient())

    store.search(query="unused PTO", vector=vector_of(0.2), use_semantic_ranker=False)

    assert "query_type" not in store._search_client().search_calls[-1]


def test_search_defaults_top_from_configuration() -> None:
    settings = AzureSearchSettings(
        endpoint="https://svc.search.windows.net", api_key=NOT_A_REAL_KEY, default_top=17
    )
    store = AzureSearchStore(settings, vector_dimensions=DIMENSIONS)
    with_search_client(store, FakeSearchClient())

    store.search(query="pto", mode=SearchMode.KEYWORD)

    assert store._search_client().search_calls[-1]["top"] == 17


def test_search_requests_only_retrieval_fields() -> None:
    store = with_search_client(azure_store(), FakeSearchClient())

    store.search(query="pto", mode=SearchMode.KEYWORD)

    selected = store._search_client().search_calls[-1]["select"]
    assert "chunk_text" in selected and "source_uri" in selected
    # The vector is thousands of floats and no caller needs it back.
    assert FIELD_EMBEDDING not in selected


def test_search_passes_composed_filters_through() -> None:
    store = with_search_client(azure_store(), FakeSearchClient())

    store.search(query="pto", mode=SearchMode.KEYWORD, document_types=["pdf"])

    assert "document_type" in store._search_client().search_calls[-1]["filter"]


def test_search_maps_rows_to_hits() -> None:
    fake = FakeSearchClient()
    row = chunk_to_document(chunk())
    row["@search.score"] = 0.9
    fake.rows = [row]
    store = with_search_client(azure_store(), fake)

    hits = store.search(query="pto", mode=SearchMode.KEYWORD)

    assert len(hits) == 1
    assert hits[0].document_id == "CNT-HR-005"
    assert hits[0].score == pytest.approx(0.9)


def test_search_failure_is_wrapped_with_the_mode() -> None:
    fake = FakeSearchClient()
    fake.search = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("429 throttled"))
    store = with_search_client(azure_store(), fake)

    with pytest.raises(IndexingError, match="hybrid search failed"):
        store.search(query="pto", vector=vector_of(0.1))


# -- argument validation, both backends ---------------------------------------


@pytest.fixture(params=["azure", "memory"])
def any_store(request) -> SearchStore:
    if request.param == "azure":
        return with_search_client(azure_store(), FakeSearchClient())
    return InMemorySearchStore(vector_dimensions=DIMENSIONS)


def test_keyword_search_requires_a_query(any_store) -> None:
    with pytest.raises(ValueError, match="requires a non-empty query"):
        any_store.search(query="   ", mode=SearchMode.KEYWORD)


def test_vector_search_requires_a_vector(any_store) -> None:
    with pytest.raises(ValueError, match="requires a vector"):
        any_store.search(mode=SearchMode.VECTOR)


def test_hybrid_search_requires_both(any_store) -> None:
    with pytest.raises(ValueError, match="requires a vector"):
        any_store.search(query="pto", mode=SearchMode.HYBRID)
    with pytest.raises(ValueError, match="requires a non-empty query"):
        any_store.search(vector=vector_of(0.1), mode=SearchMode.HYBRID)


def test_vector_width_mismatch_is_rejected_before_the_call(any_store) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        any_store.search(vector=[0.1, 0.2], mode=SearchMode.VECTOR)


# -- ranking behaviour, in-memory backend -------------------------------------


@pytest.fixture
def populated() -> InMemorySearchStore:
    store = InMemorySearchStore(vector_dimensions=DIMENSIONS)
    store.upsert(
        [
            chunk(
                0,
                document_id="CNT-HR-005",
                text="Employees may carry over five unused PTO days.",
                embedding=vector_of(1.0),
            ),
            chunk(
                0,
                document_id="CNT-SEC-009",
                text="Devices must use approved encryption controls.",
                document_type=DocumentType.DOCX,
                embedding=vector_of(-1.0),
            ),
        ]
    )
    return store


def test_keyword_search_ranks_by_term_overlap(populated) -> None:
    hits = populated.search(query="unused PTO carry over", mode=SearchMode.KEYWORD)

    assert hits[0].document_id == "CNT-HR-005"


def test_keyword_search_ignores_documents_with_no_overlap(populated) -> None:
    hits = populated.search(query="encryption", mode=SearchMode.KEYWORD)

    assert [h.document_id for h in hits] == ["CNT-SEC-009"]


def test_vector_search_ranks_by_similarity(populated) -> None:
    hits = populated.search(vector=vector_of(1.0), mode=SearchMode.VECTOR)

    assert hits[0].document_id == "CNT-HR-005"
    assert hits[-1].document_id == "CNT-SEC-009"


def test_hybrid_search_returns_both_signals(populated) -> None:
    hits = populated.search(query="encryption controls", vector=vector_of(1.0))

    # Lexical match favours SEC-009, the vector favours HR-005; fusion keeps both.
    assert {h.document_id for h in hits} == {"CNT-HR-005", "CNT-SEC-009"}


def test_search_honours_the_top_limit(populated) -> None:
    assert len(populated.search(query="the", vector=vector_of(1.0), top=1)) == 1


def test_search_filters_by_document_type(populated) -> None:
    hits = populated.search(vector=vector_of(1.0), mode=SearchMode.VECTOR, document_types=["docx"])

    assert [h.document_id for h in hits] == ["CNT-SEC-009"]


def test_search_filters_by_document_id(populated) -> None:
    hits = populated.search(vector=vector_of(1.0), mode=SearchMode.VECTOR, document_ids=["CNT-SEC-009"])

    assert [h.document_id for h in hits] == ["CNT-SEC-009"]


def test_hits_carry_full_provenance(populated) -> None:
    hit = populated.search(query="unused PTO", mode=SearchMode.KEYWORD)[0]

    assert hit.chunk_id and hit.document_id and hit.document_name
    assert hit.section and hit.section_path
    assert hit.source_uri.startswith("https://")
    assert hit.page_number == 2
    assert hit.chunk_text
