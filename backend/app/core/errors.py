"""Typed ingestion errors so the pipeline can distinguish a bad document from a
broken dependency and report accordingly."""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for every ingestion failure."""


class UnsupportedFormatError(IngestionError):
    """No loader is registered for the file extension."""


class DocumentParseError(IngestionError):
    """A loader could not read the document (corrupt, encrypted, empty)."""


class EmbeddingError(IngestionError):
    """The embedding backend failed or returned an unusable response."""


class IndexingError(IngestionError):
    """The search index rejected the write."""


class ConfigurationError(IngestionError):
    """Required configuration (endpoint, deployment, credential) is missing."""
