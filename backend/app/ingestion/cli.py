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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Probe search connectivity and index readiness, then exit. Indexes nothing.",
    )
    parser.add_argument(
        "--update-index",
        action="store_true",
        help="Apply the current schema to an existing index before ingesting.",
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

    if args.check:
        return _health_check(settings, offline=args.offline)

    root = args.source or settings.corpus_root
    try:
        documents = discover_documents(root)
        if not documents:
            logger.error("no supported documents found", extra={"source": str(root)})
            return 1

        pipeline = IngestionPipeline(settings=settings, offline=args.offline)
        report = pipeline.run(
            documents,
            force=args.force,
            dry_run=args.dry_run,
            update_index=args.update_index,
        )
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


def _health_check(settings, *, offline: bool) -> int:
    """Report search connectivity and index readiness. Exit 0 only when ready.

    Never falls back to the in-memory store implicitly: a health check that
    reports "ready" because no endpoint was configured is worse than no check at
    all. `--offline` makes that intent explicit.
    """
    from app.search import build_search_store

    if not offline and not settings.search.endpoint:
        message = (
            "AZURE_SEARCH_ENDPOINT is not set. Configure it to probe the real service, "
            "or pass --offline to check the in-memory store."
        )
        logger.error("health check cannot run", extra={"error": message})
        print(json.dumps({"ready": False, "error": message}, indent=2))
        return 1

    try:
        store = build_search_store(
            settings.search,
            vector_dimensions=settings.openai.embedding_dimensions,
            offline=offline,
        )
    except IngestionError as exc:
        logger.error("search store misconfigured", extra={"error": str(exc)})
        print(json.dumps({"ready": False, "error": str(exc)}, indent=2))
        return 1

    health = store.health()
    print(json.dumps(health.summary(), indent=2))
    return 0 if health.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
