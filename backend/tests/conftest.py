from __future__ import annotations

from pathlib import Path

import pytest

from app.core.logging import configure_logging
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.embedder import DeterministicEmbedder
from app.search import InMemorySearchStore

CORPUS = Path(__file__).resolve().parents[2] / "data"

# One representative document per supported format, used by the per-type tests.
SAMPLES = {
    "pdf": CORPUS / "pdf" / "cnt-hr-005_paid_time_off_policy.pdf",
    "docx": CORPUS / "docx" / "cnt-fin-016_expense_reimbursement_procedure.docx",
    "html": CORPUS / "html" / "cnt-hr-023_employee_separation_procedure.html",
    "markdown": CORPUS / "markdown" / "cnt-it-024_it_equipment_asset_policy.md",
}


def pytest_configure() -> None:
    configure_logging("CRITICAL", json_output=False)


@pytest.fixture(scope="session")
def corpus() -> Path:
    if not CORPUS.is_dir():
        pytest.skip(f"corpus not found at {CORPUS}")
    return CORPUS


@pytest.fixture
def sample(request, corpus: Path) -> Path:
    path = SAMPLES[request.param]
    if not path.is_file():
        pytest.skip(f"sample missing: {path}")
    return path


@pytest.fixture
def chunker() -> StructureAwareChunker:
    return StructureAwareChunker()


@pytest.fixture
def embedder() -> DeterministicEmbedder:
    # Small dimension keeps tests fast; the real deployment uses 3072.
    return DeterministicEmbedder(dimensions=32)


@pytest.fixture
def indexer() -> InMemorySearchStore:
    # Dimensions match the `embedder` fixture so vector queries validate.
    return InMemorySearchStore(vector_dimensions=32)
