"""Embedder and indexer contracts, including the Azure schema and the retry path.

The Azure clients are exercised through fakes: these tests assert the code's own
logic (batching, retry, ordering, filters, schema) without a live service.
"""

from __future__ import annotations

import math

import pytest

from app.core.config import AzureOpenAISettings, AzureSearchSettings
from app.core.errors import ConfigurationError, EmbeddingError, IndexingError
from app.ingestion.embedder import (
    AzureOpenAIEmbedder,
    DeterministicEmbedder,
    build_embedder,
)
from app.ingestion.indexer import (
    VECTOR_FIELD,
    AzureSearchIndexer,
    Indexer,
    InMemoryIndexer,
    build_indexer,
    to_search_document,
)
from app.ingestion.models import Chunk, DocumentType


def _chunk(ordinal: int = 0, *, document_id: str = "CNT-HR-005", text: str = "body") -> Chunk:
    return Chunk(
        document_id=document_id,
        document_name="Paid Time Off Policy",
        document_type=DocumentType.PDF,
        source_uri="https://blob/pto.pdf",
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


# -- embedder ------------------------------------------------------------------


def test_deterministic_embedder_is_stable_and_normalized() -> None:
    embedder = DeterministicEmbedder(dimensions=32)

    first, second, other = embedder.embed(["a", "a", "b"])

    assert first == second
    assert first != other
    assert len(first) == 32
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-6)


def test_deterministic_embedder_returns_one_vector_per_input() -> None:
    assert len(DeterministicEmbedder(8).embed(["a", "b", "c"])) == 3


def test_build_embedder_falls_back_offline_without_an_endpoint() -> None:
    embedder = build_embedder(AzureOpenAISettings(endpoint=None))

    assert isinstance(embedder, DeterministicEmbedder)


def test_azure_embedder_requires_an_endpoint() -> None:
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_ENDPOINT"):
        AzureOpenAIEmbedder(AzureOpenAISettings(endpoint=None))


class _FakeEmbeddings:
    """Stands in for `client.embeddings`, recording calls and failing on demand."""

    def __init__(self, *, fail_times: int = 0, drop_one: bool = False) -> None:
        self.batches: list[list[str]] = []
        self.fail_times = fail_times
        self.drop_one = drop_one

    def create(self, *, model: str, input: list[str], dimensions: int):
        self.batches.append(list(input))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("429 too many requests")
        items = [
            type("Item", (), {"index": i, "embedding": [float(i)] * dimensions})()
            for i in range(len(input) - (1 if self.drop_one else 0))
        ]
        # Returned out of order on purpose: the embedder must sort by index.
        items.reverse()
        usage = type("Usage", (), {"total_tokens": 7 * len(input)})()
        return type("Response", (), {"data": items, "usage": usage})()


def _embedder_with(fake, *, batch_size: int = 2) -> AzureOpenAIEmbedder:
    settings = AzureOpenAISettings(
        endpoint="https://x.openai.azure.com",
        api_key="unit-test-not-a-real-key",
        embedding_dimensions=4,
        embedding_batch_size=batch_size,
    )
    embedder = AzureOpenAIEmbedder.__new__(AzureOpenAIEmbedder)
    embedder._settings = settings
    embedder.dimensions = settings.embedding_dimensions
    embedder._client = type("Client", (), {"embeddings": fake})()
    embedder.total_tokens = 0
    return embedder


def test_azure_embedder_batches_and_restores_input_order() -> None:
    fake = _FakeEmbeddings()
    embedder = _embedder_with(fake, batch_size=2)

    vectors = embedder.embed(["a", "b", "c"])

    assert [len(b) for b in fake.batches] == [2, 1]
    assert len(vectors) == 3
    # Sorted by the API's index field, so vector n is [n, n, n, n] within a batch.
    assert vectors[0] == [0.0] * 4 and vectors[1] == [1.0] * 4
    assert embedder.total_tokens == 21


def test_azure_embedder_retries_transient_failures() -> None:
    fake = _FakeEmbeddings(fail_times=2)
    embedder = _embedder_with(fake)

    import app.ingestion.embedder as module

    original = module.time.sleep
    module.time.sleep = lambda _s: None  # keep the test fast
    try:
        vectors = embedder.embed(["a"])
    finally:
        module.time.sleep = original

    assert len(vectors) == 1
    assert len(fake.batches) == 3


def test_azure_embedder_gives_up_after_max_attempts() -> None:
    embedder = _embedder_with(_FakeEmbeddings(fail_times=99))

    import app.ingestion.embedder as module

    original = module.time.sleep
    module.time.sleep = lambda _s: None
    try:
        with pytest.raises(EmbeddingError, match="after 4 attempts"):
            embedder.embed(["a"])
    finally:
        module.time.sleep = original


def test_azure_embedder_rejects_a_short_response() -> None:
    embedder = _embedder_with(_FakeEmbeddings(drop_one=True))

    with pytest.raises(EmbeddingError, match="count mismatch"):
        embedder.embed(["a", "b"])


# -- indexer -------------------------------------------------------------------


def test_in_memory_indexer_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryIndexer(), Indexer)


def test_search_document_carries_all_required_metadata() -> None:
    chunk = _chunk()
    chunk.embedding = [0.1, 0.2]

    document = to_search_document(chunk)

    assert document["chunk_id"] == "CNT-HR-005-0000"
    assert document["document_id"] == "CNT-HR-005"
    assert document["document_name"] == "Paid Time Off Policy"
    assert document["document_type"] == "pdf"
    assert document["section"] == "Carryover"
    assert document["section_path"] == "Paid Time Off Policy > Carryover"
    assert document["page_number"] == 2
    assert document["source_uri"] == "https://blob/pto.pdf"
    assert document["content"] == "body"
    assert document[VECTOR_FIELD] == [0.1, 0.2]


def test_search_document_omits_the_vector_when_absent() -> None:
    assert VECTOR_FIELD not in to_search_document(_chunk())


def test_upsert_is_idempotent_on_repeated_writes() -> None:
    indexer = InMemoryIndexer()
    chunks = [_chunk(0), _chunk(1)]

    indexer.upsert(chunks)
    indexer.upsert(chunks)

    assert len(indexer.documents) == 2


def test_existing_hashes_reports_one_row_per_document() -> None:
    indexer = InMemoryIndexer()
    indexer.upsert([_chunk(0), _chunk(1), _chunk(0, document_id="CNT-HR-019")])

    assert indexer.existing_hashes() == {"CNT-HR-005": "hash-1", "CNT-HR-019": "hash-1"}


def test_delete_removes_only_the_named_chunks() -> None:
    indexer = InMemoryIndexer()
    indexer.upsert([_chunk(0), _chunk(1)])

    removed = indexer.delete(["CNT-HR-005-0001", "CNT-HR-005-0099"])

    assert removed == 1
    assert set(indexer.documents) == {"CNT-HR-005-0000"}


def test_azure_indexer_requires_an_endpoint() -> None:
    with pytest.raises(ConfigurationError, match="AZURE_SEARCH_ENDPOINT"):
        AzureSearchIndexer(AzureSearchSettings(endpoint=None), vector_dimensions=8)


def test_build_indexer_falls_back_offline_without_an_endpoint() -> None:
    assert isinstance(
        build_indexer(AzureSearchSettings(endpoint=None), vector_dimensions=8), InMemoryIndexer
    )


def _azure_indexer(dimensions: int = 3072) -> AzureSearchIndexer:
    return AzureSearchIndexer(
        AzureSearchSettings(
            endpoint="https://x.search.windows.net", api_key="unit-test-not-a-real-key"
        ),
        vector_dimensions=dimensions,
    )


def test_azure_index_schema_matches_the_chunk_contract() -> None:
    index = _azure_indexer(dimensions=1536)._build_index()
    fields = {f.name: f for f in index.fields}

    required = {
        "chunk_id",
        "document_id",
        "document_name",
        "document_type",
        "section",
        "section_path",
        "page_number",
        "source_uri",
        "content",
        "content_hash",
        VECTOR_FIELD,
    }
    assert required <= set(fields)
    assert fields["chunk_id"].key is True
    assert fields[VECTOR_FIELD].vector_search_dimensions == 1536
    # Hybrid retrieval needs the vector profile and the semantic config wired up.
    assert index.vector_search.profiles[0].name == fields[VECTOR_FIELD].vector_search_profile_name
    assert index.semantic_search.configurations[0].name


def test_azure_index_schema_filters_on_the_fields_retrieval_needs() -> None:
    fields = {f.name: f for f in _azure_indexer()._build_index().fields}

    for name in ("document_id", "document_type", "ordinal", "content_hash"):
        assert fields[name].filterable, f"{name} must be filterable"


def test_odata_filter_escapes_single_quotes() -> None:
    from app.ingestion.indexer import _escape_odata

    assert _escape_odata("O'Brien") == "O''Brien"


def test_upsert_raises_on_rejected_documents() -> None:
    class _RejectingClient:
        def merge_or_upload_documents(self, *, documents):
            return [
                type("R", (), {"succeeded": False, "key": "k1", "error_message": "bad field"})()
            ]

    indexer = _azure_indexer()
    indexer._search_client = lambda: _RejectingClient()

    with pytest.raises(IndexingError, match="rejected"):
        indexer.upsert([_chunk()])


def test_upsert_of_nothing_is_a_no_op() -> None:
    indexer = _azure_indexer()
    indexer._search_client = lambda: pytest.fail("must not call the service")

    assert indexer.upsert([]) == 0
    assert indexer.delete([]) == 0
