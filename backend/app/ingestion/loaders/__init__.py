"""Format loaders. Importing this package registers every loader."""

from app.ingestion.loaders.base import (
    DocumentLoader,
    loader_for,
    register,
    supported_extensions,
)
from app.ingestion.loaders.docx_loader import DocxLoader
from app.ingestion.loaders.html_loader import HtmlLoader
from app.ingestion.loaders.markdown_loader import MarkdownLoader
from app.ingestion.loaders.pdf_loader import PdfLoader

__all__ = [
    "DocumentLoader",
    "DocxLoader",
    "HtmlLoader",
    "MarkdownLoader",
    "PdfLoader",
    "loader_for",
    "register",
    "supported_extensions",
]
