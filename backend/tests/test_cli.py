"""CLI contract: exit codes and flag plumbing. The scheduler alerts on the exit
code, so it has to be right."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingestion.cli import main


@pytest.fixture
def mini_corpus(tmp_path: Path) -> Path:
    (tmp_path / "cnt-hr-900_policy.md").write_text(
        "# Policy\n\n## Carryover\n\nUp to 5 days carry over.\n", encoding="utf-8"
    )
    return tmp_path


def test_offline_run_succeeds_and_prints_a_summary(mini_corpus: Path, capsys) -> None:
    code = main(["--source", str(mini_corpus), "--offline", "--log-level", "CRITICAL"])
    summary = json.loads(capsys.readouterr().out)

    assert code == 0
    assert summary["status"] == "succeeded"
    assert summary["docs_indexed"] == 1
    assert summary["chunks_written"] == 1


def test_dry_run_reports_without_writing(mini_corpus: Path, capsys) -> None:
    code = main(["--source", str(mini_corpus), "--offline", "--dry-run", "--log-level", "CRITICAL"])
    summary = json.loads(capsys.readouterr().out)

    assert code == 0
    assert summary["docs_indexed"] == 1
    assert summary["embedding_tokens"] == 0


def test_exit_code_is_one_when_a_document_fails(tmp_path: Path, capsys) -> None:
    (tmp_path / "cnt-hr-901_broken.docx").write_bytes(b"not a docx")

    code = main(["--source", str(tmp_path), "--offline", "--log-level", "CRITICAL"])
    captured = capsys.readouterr()

    assert code == 1
    assert json.loads(captured.out)["status"] == "failed"
    assert "FAILED" in captured.err


def test_exit_code_is_one_when_nothing_is_found(tmp_path: Path) -> None:
    assert main(["--source", str(tmp_path), "--offline", "--log-level", "CRITICAL"]) == 1


def test_missing_source_is_reported_not_raised(tmp_path: Path) -> None:
    assert (
        main(["--source", str(tmp_path / "absent"), "--offline", "--log-level", "CRITICAL"]) == 1
    )
