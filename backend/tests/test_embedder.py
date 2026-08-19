"""Embedder and indexer contracts, including the Azure schema and the retry path.

The Azure clients are exercised through fakes: these tests assert the code's own
logic (batching, retry, ordering, filters, schema) without a live service.
"""

from __future__ import annotations

import math

import pytest

from app.core.config import AzureOpenAISettings
from app.core.errors import ConfigurationError, EmbeddingError
from app.ingestion.embedder import (
    AzureOpenAIEmbedder,
    DeterministicEmbedder,
    build_embedder,
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
