"""Ingestion entry point.

    python -m app.ingestion.cli --source data
    python -m app.ingestion.cli --source data --dry-run
    python -m app.ingestion.cli --source data/pdf --force

Runs as a Container Apps Job in deployment. Exit code is 1 when any document
failed, so a scheduler can alert on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import IngestionError
from app.core.logging import configure_logging, get_logger
from app.ingestion.pipeline import IngestionPipeline, discover_documents

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion.cli",
        description="Ingest the policy corpus into Azure AI Search.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="File or directory to ingest (default: CORPUS_ROOT).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed and re-index even when the content hash is unchanged.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse, normalize and chunk only. No embedding calls, no index writes.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the local embedder and in-memory index. For development only: "
        "the vectors are not semantic.",
    )
    parser.add_argument("--log-level", default=None, help="Override LOG_LEVEL.")
    parser.add_argument(
        "--plain-logs", action="store_true", help="Human-readable logs instead of JSON."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(args.log_level or settings.log_level, json_output=not args.plain_logs)

    root = args.source or settings.corpus_root
    try:
        documents = discover_documents(root)
        if not documents:
            logger.error("no supported documents found", extra={"source": str(root)})
            return 1

        pipeline = IngestionPipeline(settings=settings, offline=args.offline)
        report = pipeline.run(documents, force=args.force, dry_run=args.dry_run)
    except IngestionError as exc:
        logger.error("ingestion run aborted", extra={"error": str(exc)})
        return 1
    except KeyboardInterrupt:
        logger.warning("ingestion interrupted")
        return 130

    print(json.dumps(report.summary(), indent=2))
    for result in report.results:
        if result.outcome.value == "failed":
            print(f"FAILED  {result.source_path}: {result.error}", file=sys.stderr)
    return 1 if report.docs_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
