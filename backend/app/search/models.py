"""Data transfer objects for the search layer.

Deliberately free of Azure SDK types: these cross the boundary into services,
API schemas and tests, so nothing outside `azure_store` should ever need to
import the SDK to work with search results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SearchMode(StrEnum):
    """How a query is executed against the index."""

    KEYWORD = "keyword"  # BM25 only
    VECTOR = "vector"  # HNSW nearest-neighbour only
    HYBRID = "hybrid"  # both, fused server-side with RRF


@dataclass(slots=True)
class SearchHit:
    """One retrieved chunk, flattened from the index document."""

    chunk_id: str
    document_id: str
    document_name: str
    document_type: str
    section: str
    section_path: str
    chunk_text: str
    source_uri: str
    score: float
    page_number: int | None = None
    page_end: int | None = None
    ordinal: int | None = None
    # Populated only when the semantic ranker ran; 0-4 scale, not comparable
    # with `score`, which is an RRF or BM25 value.
    reranker_score: float | None = None

    @property
    def citation(self) -> str:
        """Human-readable provenance, e.g. `PTO Policy - Benefits > Carryover`."""
        if self.section_path and self.section_path != self.document_name:
            return f"{self.document_name} - {self.section_path}"
        return self.document_name

    @property
    def ranking_score(self) -> float:
        """The score to order by: the reranker's when present, else the fused one."""
        return self.reranker_score if self.reranker_score is not None else self.score


@dataclass(slots=True)
class DocumentIndexError:
    """One document the service refused, with the reason it gave."""

    chunk_id: str
    status_code: int | None
    error_message: str


@dataclass(slots=True)
class IndexingResult:
    """Outcome of a batch upsert or delete.

    Azure AI Search accepts a batch partially: some documents land, others are
    rejected. Reporting a single boolean would discard which is which, so both
    sides are returned and the caller decides whether a partial success is
    acceptable.
    """

    succeeded: list[str] = field(default_factory=list)
    failed: list[DocumentIndexError] = field(default_factory=list)

    @property
    def succeeded_count(self) -> int:
        return len(self.succeeded)

    @property
    def failed_count(self) -> int:
        return len(self.failed)

    @property
    def ok(self) -> bool:
        return not self.failed

    def __add__(self, other: "IndexingResult") -> "IndexingResult":
        return IndexingResult(
            succeeded=self.succeeded + other.succeeded,
            failed=self.failed + other.failed,
        )

    def summary(self) -> dict[str, object]:
        return {
            "succeeded": self.succeeded_count,
            "failed": self.failed_count,
            "first_error": self.failed[0].error_message if self.failed else None,
        }


@dataclass(slots=True)
class IndexHealth:
    """Result of a connectivity and readiness probe."""

    reachable: bool
    index_name: str
    index_exists: bool = False
    document_count: int | None = None
    vector_dimensions: int | None = None
    error: str | None = None

    @property
    def ready(self) -> bool:
        """True when the index exists and can serve queries."""
        return self.reachable and self.index_exists

    def summary(self) -> dict[str, object]:
        return {
            "reachable": self.reachable,
            "index_name": self.index_name,
            "index_exists": self.index_exists,
            "document_count": self.document_count,
            "vector_dimensions": self.vector_dimensions,
            "ready": self.ready,
            "error": self.error,
        }
