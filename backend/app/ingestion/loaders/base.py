"""Loader contract and the extension -> loader registry."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from app.core.errors import UnsupportedFormatError
from app.ingestion.models import DocumentType, RawDocument


@runtime_checkable
class DocumentLoader(Protocol):
    """Parses one source format into the common `Block` representation."""

    document_type: DocumentType
    extensions: tuple[str, ...]

    def load(self, path: Path) -> RawDocument: ...


_REGISTRY: dict[str, DocumentLoader] = {}


def register(loader: DocumentLoader) -> DocumentLoader:
    for ext in loader.extensions:
        _REGISTRY[ext.lower()] = loader
    return loader


def loader_for(path: Path) -> DocumentLoader:
    """Route by extension. Raises `UnsupportedFormatError` for unknown types."""
    try:
        return _REGISTRY[path.suffix.lower()]
    except KeyError:
        raise UnsupportedFormatError(
            f"no loader for '{path.suffix}' ({path.name}); supported: {sorted(_REGISTRY)}"
        ) from None


def supported_extensions() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
