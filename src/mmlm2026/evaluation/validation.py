from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd

# scikit-learn does not expose typed packages in this environment.
from sklearn.calibration import calibration_curve  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from xgboost import XGBClassifier


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


@dataclass(frozen=True)
class CalibrationAuditSummary:
    """Structured outputs for a raw-vs-calibrated validation audit."""

    raw_summary: ValidationSummary
    isotonic_summary: ValidationSummary
    platt_summary: ValidationSummary
    comparison: pd.DataFrame


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
    return validate_season_holdouts_with_model(
        frame,
        feature_cols=feature_cols,
        holdout_seasons=holdout_seasons,
        season_col=season_col,
        outcome_col=outcome_col,
        round_group_col=round_group_col,
        train_min_games=train_min_games,
        calibration_bins=calibration_bins,
        model_builder=build_logistic_pipeline,
    )


def validate_season_holdouts_with_model(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    holdout_seasons: list[int],
    model_builder: Callable[[list[str]], Any],
    season_col: str = "Season",
    outcome_col: str = "outcome",
    round_group_col: str = "round_group",
    train_min_games: int = 50,
    calibration_bins: int = 10,
) -> ValidationSummary:
    """Run leave-season-out validation with a supplied classifier builder."""
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

        model = model_builder(feature_cols)
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


def validate_season_holdouts_with_calibration(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    holdout_seasons: list[int],
    model_builder: Callable[[list[str]], Any],
    season_col: str = "Season",
    outcome_col: str = "outcome",
    round_group_col: str = "round_group",
    train_min_games: int = 50,
    calibration_bins: int = 10,
    clip_min: float = 0.025,
    clip_max: float = 0.975,
) -> CalibrationAuditSummary:
    """Audit raw, isotonic, and Platt-calibrated predictions on holdout seasons."""
    raw_summary = validate_season_holdouts_with_model(
        frame,
        feature_cols=feature_cols,
        holdout_seasons=holdout_seasons,
        model_builder=model_builder,
        season_col=season_col,
        outcome_col=outcome_col,
        round_group_col=round_group_col,
        train_min_games=train_min_games,
        calibration_bins=calibration_bins,
    )

    work = frame.copy()
    work = work.loc[work[outcome_col].notna()].copy()

    isotonic_rows: list[dict[str, float | int | None]] = []
    platt_rows: list[dict[str, float | int | None]] = []
    isotonic_prediction_frames: list[pd.DataFrame] = []
    platt_prediction_frames: list[pd.DataFrame] = []

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

        calibration_oof = collect_training_oof_predictions(
            train,
            feature_cols=feature_cols,
            model_builder=model_builder,
            season_col=season_col,
            outcome_col=outcome_col,
            round_group_col=round_group_col,
            train_min_games=train_min_games,
        )
        if calibration_oof.empty:
            raise ValueError(
                f"No eligible out-of-season calibration rows found before season {holdout_season}."
            )

        base_model = model_builder(feature_cols)
        base_model.fit(train[feature_cols], train[outcome_col].astype(int))
        raw_valid_preds = base_model.predict_proba(valid[feature_cols])[:, 1]

        isotonic_model = fit_probability_calibrator(
            calibration_oof["pred"],
            calibration_oof[outcome_col],
            method="isotonic",
        )
        isotonic_preds = apply_probability_calibrator(
            raw_valid_preds,
            isotonic_model,
            method="isotonic",
            clip_min=clip_min,
            clip_max=clip_max,
        )

        platt_model = fit_probability_calibrator(
            calibration_oof["pred"],
            calibration_oof[outcome_col],
            method="platt",
        )
        platt_preds = apply_probability_calibrator(
            raw_valid_preds,
            platt_model,
            method="platt",
            clip_min=clip_min,
            clip_max=clip_max,
        )

        isotonic_prediction_frames.append(
            _build_prediction_frame(
                valid,
                preds=isotonic_preds,
                season_col=season_col,
                outcome_col=outcome_col,
                round_group_col=round_group_col,
            )
        )
        platt_prediction_frames.append(
            _build_prediction_frame(
                valid,
                preds=platt_preds,
                season_col=season_col,
                outcome_col=outcome_col,
                round_group_col=round_group_col,
            )
        )
        isotonic_rows.append(
            _build_metrics_row(
                valid,
                preds=isotonic_preds,
                season=holdout_season,
                outcome_col=outcome_col,
                round_group_col=round_group_col,
                train_games=train_games,
            )
        )
        platt_rows.append(
            _build_metrics_row(
                valid,
                preds=platt_preds,
                season=holdout_season,
                outcome_col=outcome_col,
                round_group_col=round_group_col,
                train_games=train_games,
            )
        )

    isotonic_summary = _build_validation_summary_from_rows(
        isotonic_rows,
        isotonic_prediction_frames,
        outcome_col=outcome_col,
        calibration_bins=calibration_bins,
    )
    platt_summary = _build_validation_summary_from_rows(
        platt_rows,
        platt_prediction_frames,
        outcome_col=outcome_col,
        calibration_bins=calibration_bins,
    )

    comparison = (
        pd.DataFrame(
            [
                _comparison_row("raw", raw_summary),
                _comparison_row("isotonic", isotonic_summary),
                _comparison_row("platt", platt_summary),
            ]
        )
        .sort_values("flat_brier")
        .reset_index(drop=True)
    )

    return CalibrationAuditSummary(
        raw_summary=raw_summary,
        isotonic_summary=isotonic_summary,
        platt_summary=platt_summary,
        comparison=comparison,
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


def collect_training_oof_predictions(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    model_builder: Callable[[list[str]], Any],
    season_col: str = "Season",
    outcome_col: str = "outcome",
    round_group_col: str = "round_group",
    train_min_games: int = 50,
) -> pd.DataFrame:
    """Collect out-of-season predictions from training seasons for calibration."""
    work = frame.loc[frame[outcome_col].notna()].copy()
    seasons = sorted(int(season) for season in work[season_col].unique())
    prediction_frames: list[pd.DataFrame] = []

    for calib_season in seasons:
        train = work.loc[work[season_col] < calib_season].copy()
        calib = work.loc[work[season_col] == calib_season].copy()
        if calib.empty or len(train) < train_min_games:
            continue

        model = model_builder(feature_cols)
        model.fit(train[feature_cols], train[outcome_col].astype(int))
        preds = model.predict_proba(calib[feature_cols])[:, 1]
        prediction_frames.append(
            _build_prediction_frame(
                calib,
                preds=preds,
                season_col=season_col,
                outcome_col=outcome_col,
                round_group_col=round_group_col,
            )
        )

    if not prediction_frames:
        return pd.DataFrame(columns=[season_col, outcome_col, "pred"])
    return pd.concat(prediction_frames, ignore_index=True)


def fit_probability_calibrator(
    y_prob: pd.Series | list[float],
    y_true: pd.Series | list[int],
    *,
    method: str,
) -> Any:
    """Fit a probability calibrator on out-of-fold predictions."""
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(y_prob, y_true)
        return model
    if method == "platt":
        model = LogisticRegression(max_iter=1_000)
        model.fit(pd.Series(y_prob).to_numpy().reshape(-1, 1), y_true)
        return model
    raise ValueError(f"Unsupported calibration method: {method}")


def apply_probability_calibrator(
    y_prob: pd.Series | list[float],
    calibrator: Any,
    *,
    method: str,
    clip_min: float = 0.025,
    clip_max: float = 0.975,
) -> pd.Series:
    """Apply a fitted calibrator and clip calibrated probabilities."""
    prob_series = pd.Series(y_prob, dtype=float)
    if method == "isotonic":
        calibrated = pd.Series(calibrator.predict(prob_series), index=prob_series.index)
    elif method == "platt":
        calibrated = pd.Series(
            calibrator.predict_proba(prob_series.to_numpy().reshape(-1, 1))[:, 1],
            index=prob_series.index,
        )
    else:
        raise ValueError(f"Unsupported calibration method: {method}")
    return calibrated.clip(lower=clip_min, upper=clip_max)


def expected_calibration_error(calibration_table: pd.DataFrame) -> float:
    """Return expected calibration error from a calibration table."""
    if calibration_table.empty:
        return 0.0
    total = int(calibration_table["count"].sum())
    if total == 0:
        return 0.0
    weighted_gap = (calibration_table["absolute_gap"] * calibration_table["count"]).sum()
    return float(weighted_gap / total)


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


def build_hist_gbt_pipeline(feature_cols: list[str]) -> Pipeline:
    """Build a histogram gradient boosting pipeline for numeric matchup features."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                feature_cols,
            )
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=3,
                    max_iter=200,
                    min_samples_leaf=10,
                    random_state=0,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(feature_cols: list[str]) -> Pipeline:
    """Build an XGBoost classifier pipeline for numeric matchup features."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                feature_cols,
            )
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    min_child_weight=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=0,
                ),
            ),
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
    pred_series = pd.Series(pd.Series(preds, dtype=float).to_numpy(), index=frame.index)
    return float(brier_score_loss(frame.loc[mask, outcome_col], pred_series.loc[mask]))


def _build_prediction_frame(
    valid: pd.DataFrame,
    *,
    preds: pd.Series | list[float] | Any,
    season_col: str,
    outcome_col: str,
    round_group_col: str,
) -> pd.DataFrame:
    keep_cols = [season_col, outcome_col]
    for optional_col in ["LowTeamID", "HighTeamID"]:
        if optional_col in valid.columns:
            keep_cols.append(optional_col)
    pred_frame = valid[keep_cols].copy()
    pred_frame["pred"] = pd.Series(preds, dtype=float).to_numpy()
    if round_group_col in valid.columns:
        pred_frame[round_group_col] = valid[round_group_col]
    return pred_frame


def _build_metrics_row(
    valid: pd.DataFrame,
    *,
    preds: pd.Series | list[float],
    season: int,
    outcome_col: str,
    round_group_col: str,
    train_games: int,
) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {
        "Season": season,
        "train_tourney_games": train_games,
        "valid_tourney_games": len(valid),
        "flat_brier": float(brier_score_loss(valid[outcome_col], preds)),
        "log_loss": float(log_loss(valid[outcome_col], preds, labels=[0, 1])),
    }
    if round_group_col in valid.columns:
        row["r1_brier"] = _group_brier(valid, preds, outcome_col, round_group_col, "R1")
        row["r2plus_brier"] = _group_brier(valid, preds, outcome_col, round_group_col, "R2+")
    return row


def _build_validation_summary_from_rows(
    metrics_rows: list[dict[str, float | int | None]],
    prediction_frames: list[pd.DataFrame],
    *,
    outcome_col: str,
    calibration_bins: int,
) -> ValidationSummary:
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


def _comparison_row(method: str, summary: ValidationSummary) -> dict[str, float | str]:
    return {
        "method": method,
        "flat_brier": summary.overall_flat_brier,
        "log_loss": summary.overall_log_loss,
        "ece": expected_calibration_error(summary.calibration_table),
        "max_abs_gap": float(summary.calibration_table["absolute_gap"].max()),
    }


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
