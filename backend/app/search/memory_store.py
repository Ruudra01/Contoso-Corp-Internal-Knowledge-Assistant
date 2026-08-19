"""In-process `SearchStore` for tests, offline runs and local development.

Implements the same contract as the Azure store, including keyword, vector and
hybrid ranking, so pipeline and retrieval logic can be exercised end to end with
no service and no credentials.

The ranking is deliberately simple — term overlap for keyword, cosine for
vector, Reciprocal Rank Fusion to combine them. It reproduces the *shape* of the
Azure behaviour (ordering, fusion, filtering), not its scores, so it must never
be used to serve real retrieval.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

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
    FIELD_CHUNK_TEXT,
    FIELD_CONTENT_HASH,
    FIELD_DOCUMENT_ID,
    FIELD_DOCUMENT_NAME,
    FIELD_DOCUMENT_TYPE,
    FIELD_EMBEDDING,
    FIELD_ORDINAL,
    FIELD_SECTION,
    FIELD_SECTION_PATH,
    chunk_to_document,
    document_to_hit,
)

logger = get_logger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
# RRF constant; 60 is the value Azure AI Search documents for its own fusion.
_RRF_K = 60
# Fields term overlap is scored against, mirroring the searchable set.
_SEARCHABLE = (FIELD_CHUNK_TEXT, FIELD_DOCUMENT_NAME, FIELD_SECTION, FIELD_SECTION_PATH)


class InMemorySearchStore:
    """Dictionary-backed index with Azure-shaped semantics."""

    def __init__(self, *, vector_dimensions: int = 3072, index_name: str = "in-memory") -> None:
        self.index_name = index_name
        self.vector_dimensions = vector_dimensions
        self.documents: dict[str, dict[str, Any]] = {}
        # Test affordances.
        self.ensure_index_calls = 0
        self.deleted: list[str] = []
        # Keys that should be rejected, to exercise partial-failure handling.
        self.reject_keys: set[str] = set()

    # -- schema -------------------------------------------------------------

    def ensure_index(self, *, allow_update: bool = False) -> bool:
        self.ensure_index_calls += 1
        return True

    def health(self) -> IndexHealth:
        return IndexHealth(
            reachable=True,
            index_name=self.index_name,
            index_exists=True,
            document_count=len(self.documents),
            vector_dimensions=self.vector_dimensions,
        )

    # -- writes -------------------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk]) -> IndexingResult:
        result = IndexingResult()
        for chunk in chunks:
            if chunk.chunk_id in self.reject_keys:
                result.failed.append(
                    DocumentIndexError(
                        chunk_id=chunk.chunk_id,
                        status_code=400,
                        error_message="rejected by test fixture",
                    )
                )
                continue
            self.documents[chunk.chunk_id] = chunk_to_document(chunk)
            result.succeeded.append(chunk.chunk_id)
        return result

    def delete(self, chunk_ids: Iterable[str]) -> IndexingResult:
        result = IndexingResult()
        for chunk_id in chunk_ids:
            if self.documents.pop(chunk_id, None) is not None:
                self.deleted.append(chunk_id)
                result.succeeded.append(chunk_id)
            else:
                # Azure treats deleting an absent key as a success, not an error.
                result.succeeded.append(chunk_id)
        return result

    def delete_document(self, document_id: str) -> IndexingResult:
        return self.delete(sorted(self.chunk_ids_for(document_id)))

    # -- reads --------------------------------------------------------------

    def existing_hashes(self) -> dict[str, str]:
        return {
            document[FIELD_DOCUMENT_ID]: document[FIELD_CONTENT_HASH]
            for document in self.documents.values()
            if document.get(FIELD_ORDINAL) == 0 and document.get(FIELD_CONTENT_HASH)
        }

    def chunk_ids_for(self, document_id: str) -> set[str]:
        return {
            key
            for key, document in self.documents.items()
            if document.get(FIELD_DOCUMENT_ID) == document_id
        }

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
        mode = SearchMode(mode)
        if mode in (SearchMode.KEYWORD, SearchMode.HYBRID) and not (query and query.strip()):
            raise ValueError(f"{mode} search requires a non-empty query")
        if mode in (SearchMode.VECTOR, SearchMode.HYBRID) and vector is None:
            raise ValueError(f"{mode} search requires a vector")
        if vector is not None and len(vector) != self.vector_dimensions:
            raise ValueError(
                f"vector has {len(vector)} dimensions, index expects {self.vector_dimensions}"
            )

        candidates = [
            document
            for document in self.documents.values()
            if self._passes(document, document_ids, document_types)
        ]
        size = top or 30

        if mode is SearchMode.KEYWORD:
            ranked = self._by_keyword(candidates, query or "")
        elif mode is SearchMode.VECTOR:
            ranked = self._by_vector(candidates, vector or [])
        else:
            ranked = self._fuse(
                self._by_keyword(candidates, query or ""),
                self._by_vector(candidates, vector or []),
            )

        hits: list[SearchHit] = []
        for document, score in ranked[:size]:
            hit = document_to_hit(document)
            hit.score = score
            hits.append(hit)
        return hits

    # -- ranking ------------------------------------------------------------

    @staticmethod
    def _passes(
        document: dict[str, Any],
        document_ids: Sequence[str] | None,
        document_types: Sequence[str] | None,
    ) -> bool:
        if document_ids and document.get(FIELD_DOCUMENT_ID) not in set(document_ids):
            return False
        if document_types and document.get(FIELD_DOCUMENT_TYPE) not in set(document_types):
            return False
        return True

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(_WORD.findall(text.lower()))

    def _by_keyword(
        self, documents: list[dict[str, Any]], query: str
    ) -> list[tuple[dict[str, Any], float]]:
        """Term-overlap ranking over the searchable fields."""
        query_terms = self._terms(query)
        if not query_terms:
            return [(d, 0.0) for d in documents]
        scored: list[tuple[dict[str, Any], float]] = []
        for document in documents:
            haystack = " ".join(str(document.get(f, "")) for f in _SEARCHABLE)
            overlap = len(query_terms & self._terms(haystack))
            if overlap:
                scored.append((document, overlap / len(query_terms)))
        scored.sort(key=lambda pair: (-pair[1], pair[0].get(FIELD_CHUNK_ID, "")))
        return scored

    def _by_vector(
        self, documents: list[dict[str, Any]], vector: Sequence[float]
    ) -> list[tuple[dict[str, Any], float]]:
        """Cosine similarity against the stored embeddings."""
        scored: list[tuple[dict[str, Any], float]] = []
        for document in documents:
            stored = document.get(FIELD_EMBEDDING)
            if not stored:
                continue
            scored.append((document, _cosine(vector, stored)))
        scored.sort(key=lambda pair: (-pair[1], pair[0].get(FIELD_CHUNK_ID, "")))
        return scored

    @staticmethod
    def _fuse(
        keyword: list[tuple[dict[str, Any], float]],
        vector: list[tuple[dict[str, Any], float]],
    ) -> list[tuple[dict[str, Any], float]]:
        """Reciprocal Rank Fusion, the same combination Azure applies to hybrid.

        Rank-based rather than score-based, because BM25 and cosine values are
        not on a comparable scale.
        """
        fused: dict[str, float] = {}
        documents: dict[str, dict[str, Any]] = {}
        for ranking in (keyword, vector):
            for rank, (document, _score) in enumerate(ranking, start=1):
                key = document.get(FIELD_CHUNK_ID, "")
                fused[key] = fused.get(key, 0.0) + 1.0 / (_RRF_K + rank)
                documents[key] = document
        ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return [(documents[key], score) for key, score in ordered]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
