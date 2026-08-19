"""The search abstraction the rest of the application depends on.

Nothing above this layer imports `azure.search.documents`. Services, the API and
the ingestion pipeline talk to `SearchStore`; swapping Azure AI Search for
another engine means adding one implementation, not touching callers.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence, runtime_checkable

from app.ingestion.models import Chunk
from app.search.models import IndexHealth, IndexingResult, SearchHit, SearchMode


@runtime_checkable
class SearchStore(Protocol):
    """Write and read access to the chunk index."""

    index_name: str
    vector_dimensions: int

    # -- schema -------------------------------------------------------------

    def ensure_index(self, *, allow_update: bool = False) -> bool:
        """Create the index when absent.

        Returns True when the index was created or updated. With
        `allow_update=False` an existing index is left untouched.
        """
        ...

    def health(self) -> IndexHealth:
        """Probe connectivity and index readiness. Never raises."""
        ...

    # -- writes -------------------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk]) -> IndexingResult:
        """Insert or replace chunks by key, in batches. Partial failures are
        reported, not raised."""
        ...

    def delete(self, chunk_ids: Iterable[str]) -> IndexingResult:
        """Delete by key. Missing keys are not an error."""
        ...

    def delete_document(self, document_id: str) -> IndexingResult:
        """Delete every chunk belonging to one document."""
        ...

    # -- reads --------------------------------------------------------------

    def existing_hashes(self) -> dict[str, str]:
        """`{document_id: content_hash}` for already-indexed documents."""
        ...

    def chunk_ids_for(self, document_id: str) -> set[str]:
        """Every chunk key currently stored for one document."""
        ...

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
        """Run a keyword, vector or hybrid query and return ranked hits."""
        ...
