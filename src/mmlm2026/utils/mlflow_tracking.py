from __future__ import annotations

import subprocess
from typing import Any

import mlflow


def _git_value(args: list[str], default: str = "unknown") -> str:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return default
    return result.stdout.strip() or default


def git_context() -> dict[str, str]:
    """Return basic git context for experiment tags."""
    return {
        "git_sha": _git_value(["git", "rev-parse", "HEAD"]),
        "git_branch": _git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def start_tracked_run(
    run_name: str,
    tags: dict[str, Any] | None = None,
) -> mlflow.ActiveRun:
    """Start an MLflow run and attach standard git tags."""
    run = mlflow.start_run(run_name=run_name)
    merged_tags: dict[str, Any] = {**git_context(), **(tags or {})}
    mlflow.set_tags(merged_tags)
    return run
