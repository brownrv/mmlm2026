from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mmlm2026.evaluation.validation import (
    CalibrationAuditSummary,
    ValidationSummary,
    apply_probability_calibrator,
    build_logistic_pipeline,
    fit_probability_calibrator,
    save_validation_artifacts,
    validate_season_holdouts,
    validate_season_holdouts_with_calibration,
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


def test_validate_season_holdouts_with_calibration_returns_comparison() -> None:
    frame = _build_validation_frame()

    audit = validate_season_holdouts_with_calibration(
        frame,
        feature_cols=["seed_diff", "elo_diff"],
        holdout_seasons=[2023, 2024],
        model_builder=build_logistic_pipeline,
        train_min_games=50,
    )

    assert isinstance(audit, CalibrationAuditSummary)
    assert audit.comparison["method"].tolist() == ["raw", "isotonic", "platt"] or set(
        audit.comparison["method"]
    ) == {"raw", "isotonic", "platt"}
    assert {"flat_brier", "log_loss", "ece", "max_abs_gap"}.issubset(audit.comparison.columns)


def test_probability_calibrator_clips_outputs() -> None:
    raw_preds = pd.Series([0.01, 0.2, 0.8, 0.99])
    outcomes = pd.Series([0, 0, 1, 1])

    isotonic = fit_probability_calibrator(raw_preds, outcomes, method="isotonic")
    calibrated = apply_probability_calibrator(
        raw_preds,
        isotonic,
        method="isotonic",
        clip_min=0.1,
        clip_max=0.9,
    )

    assert calibrated.min() >= 0.1
    assert calibrated.max() <= 0.9
