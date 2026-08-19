"""Embedding generation.

`Embedder` is the seam the pipeline depends on. `AzureOpenAIEmbedder` is the
production path; `DeterministicEmbedder` lets the pipeline, the chunker and the
indexer be tested end to end with no network and no credentials.
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from typing import Protocol, Sequence, runtime_checkable

from app.core.config import AzureOpenAISettings
from app.core.errors import ConfigurationError, EmbeddingError
from app.core.logging import get_logger
from app.ingestion.tokens import count_tokens, truncate_to_tokens

logger = get_logger(__name__)

# `text-embedding-3-*` rejects inputs beyond 8191 tokens.
_MODEL_TOKEN_LIMIT = 8191
_MAX_ATTEMPTS = 4
_BACKOFF_SECONDS = 1.5


@runtime_checkable
class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input, in input order."""
        ...


class AzureOpenAIEmbedder:
    """Azure OpenAI embeddings, batched, with retry on transient failures.

    Auth: an API key from configuration when present, otherwise
    `DefaultAzureCredential` (Managed Identity in Container Apps). No credential
    is ever read from source.
    """

    def __init__(self, settings: AzureOpenAISettings) -> None:
        if not settings.endpoint:
            raise ConfigurationError(
                "AZURE_OPENAI_ENDPOINT is not set; export it or use DeterministicEmbedder for offline runs"
            )
        self._settings = settings
        self.dimensions = settings.embedding_dimensions
        self._client = self._build_client(settings)
        self.total_tokens = 0

    @staticmethod
    def _build_client(settings: AzureOpenAISettings):
        from openai import AzureOpenAI

        if settings.api_key:
            logger.info("embedder authenticating with API key from configuration")
            return AzureOpenAI(
                azure_endpoint=settings.endpoint,
                api_key=settings.api_key,
                api_version=settings.api_version,
            )

        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        logger.info("embedder authenticating with DefaultAzureCredential")
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return AzureOpenAI(
            azure_endpoint=settings.endpoint,
            azure_ad_token_provider=token_provider,
            api_version=settings.api_version,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            batch = [self._fit(t) for t in texts[start : start + batch_size]]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _fit(self, text: str) -> str:
        """Keep a single input inside the model's token ceiling.

        The chunker's `max_tokens` is far below this, so truncation here only
        fires for a pathological atomic block that could not be split.
        """
        if count_tokens(text) <= _MODEL_TOKEN_LIMIT:
            return text
        logger.warning("truncating oversized embedding input", extra={"limit": _MODEL_TOKEN_LIMIT})
        return truncate_to_tokens(text, _MODEL_TOKEN_LIMIT)

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.embeddings.create(
                    model=self._settings.embedding_deployment,
                    input=batch,
                    dimensions=self.dimensions,
                )
            except Exception as exc:
                last_error = exc
                if attempt == _MAX_ATTEMPTS:
                    break
                delay = _BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "embedding batch failed, retrying",
                    extra={"attempt": attempt, "delay_s": delay, "error": str(exc)},
                )
                time.sleep(delay)
                continue

            if usage := getattr(response, "usage", None):
                self.total_tokens += getattr(usage, "total_tokens", 0) or 0
            # The API documents input order preservation, but index is authoritative.
            ordered = sorted(response.data, key=lambda d: d.index)
            if len(ordered) != len(batch):
                raise EmbeddingError(
                    f"embedding count mismatch: sent {len(batch)}, received {len(ordered)}"
                )
            return [list(item.embedding) for item in ordered]

        raise EmbeddingError(f"embedding failed after {_MAX_ATTEMPTS} attempts: {last_error}")


class DeterministicEmbedder:
    """Hash-based pseudo-embeddings for offline runs and tests.

    Same text always yields the same unit vector, and different text yields a
    different one, which is all the pipeline and index tests need. It carries no
    semantic meaning, so it must never be used to serve retrieval.
    """

    def __init__(self, dimensions: int = 3072) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.total_tokens = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        self.total_tokens += count_tokens(text)
        values: list[float] = []
        counter = 0
        while len(values) < self.dimensions:
            digest = hashlib.sha256(f"{text}|{counter}".encode("utf-8")).digest()
            # 8 float32s per 32-byte digest.
            values.extend(struct.unpack("<8f", digest))
            counter += 1
        values = values[: self.dimensions]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


def build_embedder(settings: AzureOpenAISettings, *, offline: bool = False) -> Embedder:
    """Pick a backend. `offline` (or a missing endpoint) selects the local one."""
    if offline or not settings.endpoint:
        logger.warning(
            "using DeterministicEmbedder: vectors are not semantic and must not serve retrieval",
            extra={"reason": "offline flag" if offline else "AZURE_OPENAI_ENDPOINT unset"},
        )
        return DeterministicEmbedder(settings.embedding_dimensions)
    return AzureOpenAIEmbedder(settings)
