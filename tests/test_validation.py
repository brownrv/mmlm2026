from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mmlm2026.evaluation.validation import (
    ValidationSummary,
    save_validation_artifacts,
    validate_season_holdouts,
)


def _build_validation_frame() -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    for season in range(2010, 2025):
        for game_idx in range(60):
            seed_diff = ((game_idx % 8) - 3) / 3
            elo_diff = seed_diff * 40 + (game_idx % 5) * 3
            outcome = 1 if (seed_diff + (game_idx % 3) * 0.05) > 0 else 0
            rows.append(
                {
                    "Season": season,
                    "seed_diff": float(seed_diff),
                    "elo_diff": float(elo_diff),
                    "outcome": outcome,
                    "round_group": "R1" if game_idx < 30 else "R2+",
                }
            )
    return pd.DataFrame(rows)


def test_validate_season_holdouts_returns_metrics_and_predictions() -> None:
    frame = _build_validation_frame()

    summary = validate_season_holdouts(
        frame,
        feature_cols=["seed_diff", "elo_diff"],
        holdout_seasons=[2023, 2024],
        train_min_games=50,
    )

    assert isinstance(summary, ValidationSummary)
    assert summary.per_season_metrics["Season"].tolist() == [2023, 2024]
    assert {"flat_brier", "log_loss", "r1_brier", "r2plus_brier"}.issubset(
        summary.per_season_metrics.columns
    )
    assert set(summary.oof_predictions["Season"]) == {2023, 2024}
    assert summary.overall_flat_brier >= 0.0


def test_validate_season_holdouts_requires_enough_training_games() -> None:
    frame = _build_validation_frame()

    with pytest.raises(ValueError, match="minimum required is 50"):
        validate_season_holdouts(
            frame,
            feature_cols=["seed_diff", "elo_diff"],
            holdout_seasons=[2010],
            train_min_games=50,
        )


def test_save_validation_artifacts_writes_expected_files(tmp_path: Path) -> None:
    frame = _build_validation_frame()
    summary = validate_season_holdouts(
        frame,
        feature_cols=["seed_diff", "elo_diff"],
        holdout_seasons=[2024],
        train_min_games=50,
    )

    artifacts = save_validation_artifacts(summary, output_dir=tmp_path)

    assert artifacts.per_season_metrics_path.exists()
    assert artifacts.calibration_table_path.exists()
    assert artifacts.oof_predictions_path.exists()
    assert artifacts.reliability_diagram_path.exists()
