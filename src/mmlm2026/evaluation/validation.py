from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# scikit-learn does not expose typed packages in this environment.
from sklearn.calibration import calibration_curve  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]


@dataclass(frozen=True)
class ValidationArtifacts:
    """Paths to materialized validation artifacts."""

    per_season_metrics_path: Path
    calibration_table_path: Path
    oof_predictions_path: Path
    reliability_diagram_path: Path


@dataclass(frozen=True)
class ValidationSummary:
    """Structured outputs for a leave-season-out validation run."""

    per_season_metrics: pd.DataFrame
    calibration_table: pd.DataFrame
    oof_predictions: pd.DataFrame

    @property
    def overall_flat_brier(self) -> float:
        """Return flat Brier on all out-of-fold predictions."""
        return float(
            brier_score_loss(
                self.oof_predictions["outcome"],
                self.oof_predictions["pred"],
            )
        )

    @property
    def overall_log_loss(self) -> float:
        """Return log-loss on all out-of-fold predictions."""
        return float(
            log_loss(
                self.oof_predictions["outcome"],
                self.oof_predictions["pred"],
                labels=[0, 1],
            )
        )


def validate_season_holdouts(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    holdout_seasons: list[int],
    season_col: str = "Season",
    outcome_col: str = "outcome",
    round_group_col: str = "round_group",
    train_min_games: int = 50,
    calibration_bins: int = 10,
) -> ValidationSummary:
    """Run leave-season-out validation with strict pre-holdout training data."""
    required = {season_col, outcome_col, *feature_cols}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")

    if not holdout_seasons:
        raise ValueError("At least one holdout season is required.")

    work = frame.copy()
    work = work.loc[work[outcome_col].notna()].copy()

    metrics_rows: list[dict[str, float | int | None]] = []
    prediction_frames: list[pd.DataFrame] = []

    for holdout_season in holdout_seasons:
        train = work.loc[work[season_col] < holdout_season].copy()
        valid = work.loc[work[season_col] == holdout_season].copy()

        if valid.empty:
            raise ValueError(f"No validation rows found for season {holdout_season}.")

        train_games = len(train)
        if train_games < train_min_games:
            raise ValueError(
                f"Training set for season {holdout_season} has {train_games} games; "
                f"minimum required is {train_min_games}."
            )

        model = build_logistic_pipeline(feature_cols)
        model.fit(train[feature_cols], train[outcome_col].astype(int))

        preds = model.predict_proba(valid[feature_cols])[:, 1]
        keep_cols = [season_col, outcome_col]
        for optional_col in ["LowTeamID", "HighTeamID"]:
            if optional_col in valid.columns:
                keep_cols.append(optional_col)
        pred_frame = valid[keep_cols].copy()
        pred_frame["pred"] = preds
        if round_group_col in valid.columns:
            pred_frame[round_group_col] = valid[round_group_col]
        prediction_frames.append(pred_frame)

        row: dict[str, float | int | None] = {
            "Season": holdout_season,
            "train_tourney_games": train_games,
            "valid_tourney_games": len(valid),
            "flat_brier": float(brier_score_loss(valid[outcome_col], preds)),
            "log_loss": float(log_loss(valid[outcome_col], preds, labels=[0, 1])),
        }

        if round_group_col in valid.columns:
            row["r1_brier"] = _group_brier(valid, preds, outcome_col, round_group_col, "R1")
            row["r2plus_brier"] = _group_brier(valid, preds, outcome_col, round_group_col, "R2+")

        metrics_rows.append(row)

    per_season_metrics = pd.DataFrame(metrics_rows).sort_values("Season").reset_index(drop=True)
    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    calibration_table = build_calibration_table(
        oof_predictions[outcome_col].astype(int),
        oof_predictions["pred"],
        n_bins=calibration_bins,
    )

    return ValidationSummary(
        per_season_metrics=per_season_metrics,
        calibration_table=calibration_table,
        oof_predictions=oof_predictions,
    )


def build_calibration_table(
    y_true: pd.Series,
    y_prob: pd.Series,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a calibration summary table for reliability analysis."""
    observed_rate, predicted_mean = calibration_curve(
        y_true.astype(int),
        y_prob.astype(float),
        n_bins=n_bins,
        strategy="uniform",
    )

    bins = pd.cut(
        y_prob.astype(float),
        bins=n_bins,
        labels=False,
        include_lowest=True,
    )
    bin_counts = bins.value_counts().sort_index()

    rows: list[dict[str, float | int]] = []
    for idx, (obs, pred) in enumerate(zip(observed_rate, predicted_mean, strict=False)):
        rows.append(
            {
                "bin_index": idx,
                "predicted_mean": float(pred),
                "observed_rate": float(obs),
                "absolute_gap": float(abs(obs - pred)),
                "count": int(bin_counts.iloc[idx]) if idx < len(bin_counts) else 0,
            }
        )

    return pd.DataFrame(rows)


def save_validation_artifacts(
    summary: ValidationSummary,
    *,
    output_dir: str | Path,
) -> ValidationArtifacts:
    """Persist validation metrics and plots to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_season_metrics_path = out_dir / "per_season_metrics.csv"
    calibration_table_path = out_dir / "calibration_table.csv"
    oof_predictions_path = out_dir / "oof_predictions.csv"
    reliability_diagram_path = out_dir / "reliability_diagram.png"

    summary.per_season_metrics.to_csv(per_season_metrics_path, index=False)
    summary.calibration_table.to_csv(calibration_table_path, index=False)
    summary.oof_predictions.to_csv(oof_predictions_path, index=False)
    _save_reliability_diagram(summary.calibration_table, reliability_diagram_path)

    return ValidationArtifacts(
        per_season_metrics_path=per_season_metrics_path,
        calibration_table_path=calibration_table_path,
        oof_predictions_path=oof_predictions_path,
        reliability_diagram_path=reliability_diagram_path,
    )


def build_logistic_pipeline(feature_cols: list[str]) -> Pipeline:
    """Build the standard logistic pipeline used by validation and baselines."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1_000)),
        ]
    )


def _group_brier(
    frame: pd.DataFrame,
    preds: pd.Series | list[float],
    outcome_col: str,
    round_group_col: str,
    group_value: str,
) -> float | None:
    mask = frame[round_group_col] == group_value
    if not mask.any():
        return None
    pred_series = pd.Series(preds, index=frame.index)
    return float(brier_score_loss(frame.loc[mask, outcome_col], pred_series.loc[mask]))


def _save_reliability_diagram(calibration_table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(
        calibration_table["predicted_mean"],
        calibration_table["observed_rate"],
        marker="o",
        label="Model",
    )
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
