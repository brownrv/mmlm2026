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
    """Start an MLflow run and attach standard git context tags.

    Recommended tags to pass (see AGENTS.md for full list):
        - hypothesis: one-sentence description of what you expect to learn
        - model_family: e.g. "logistic_regression", "lgbm", "ensemble"
        - league: "men", "women", or "combined"
        - season_window: e.g. "2010-2025"
        - depends_on: e.g. "data:processed_v1", "feature:seed_diff_v2"
        - retest_if: condition that would invalidate this experiment

    Example::

        with start_tracked_run(
            "seed-diff-baseline-v1",
            tags={
                "hypothesis": "seed difference alone predicts win probability well",
                "model_family": "logistic_regression",
                "league": "men",
                "season_window": "2010-2025",
                "depends_on": "data:processed_v1",
                "retest_if": "seed assignment rules change",
            },
        ):
            mlflow.log_params({"feature": "seed_diff", "regularization": "l2"})
            mlflow.log_metrics({"brier_score": 0.21})
    """
    run = mlflow.start_run(run_name=run_name)
    merged_tags: dict[str, Any] = {**git_context(), **(tags or {})}
    mlflow.set_tags(merged_tags)
    return run
