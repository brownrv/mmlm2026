from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]


@dataclass(frozen=True)
class MetaTuningResult:
    alpha: float
    per_alpha: pd.DataFrame


def logit_clip(probabilities: pd.Series | np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    return np.log(probs / (1.0 - probs))


def fit_logit_ridge_meta(
    train_frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    alpha: float,
) -> Ridge:
    model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
    model.fit(train_frame[feature_cols], train_frame["outcome"].astype(float))
    return model


def score_logit_ridge_meta(
    model: Ridge,
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
) -> pd.Series:
    raw = pd.Series(model.predict(frame[feature_cols]), index=frame.index, dtype=float)
    return raw.clip(lower=1e-6, upper=1.0 - 1e-6)


def tune_logit_ridge_alpha(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    season_col: str = "Season",
    alpha_grid: list[float] | None = None,
) -> MetaTuningResult:
    if alpha_grid is None:
        alpha_grid = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    seasons = sorted(int(season) for season in frame[season_col].unique())
    rows: list[dict[str, float]] = []
    for alpha in alpha_grid:
        preds: list[pd.DataFrame] = []
        for holdout_season in seasons:
            train = frame.loc[frame[season_col] < holdout_season].copy()
            valid = frame.loc[frame[season_col] == holdout_season].copy()
            if train.empty or valid.empty:
                continue
            model = fit_logit_ridge_meta(train, feature_cols=feature_cols, alpha=float(alpha))
            pred = score_logit_ridge_meta(model, valid, feature_cols=feature_cols)
            preds.append(
                pd.DataFrame(
                    {
                        "Season": valid[season_col].to_numpy(),
                        "outcome": valid["outcome"].astype(int).to_numpy(),
                        "pred": pred.to_numpy(),
                    }
                )
            )
        if not preds:
            continue
        merged = pd.concat(preds, ignore_index=True)
        rows.append(
            {
                "alpha": float(alpha),
                "flat_brier": float(brier_score_loss(merged["outcome"], merged["pred"])),
                "log_loss": float(log_loss(merged["outcome"], merged["pred"])),
            }
        )

    if not rows:
        raise ValueError("No rows available to tune the logit-Ridge meta-learner.")

    grid = pd.DataFrame(rows).sort_values(["flat_brier", "alpha"]).reset_index(drop=True)
    return MetaTuningResult(alpha=float(grid.iloc[0]["alpha"]), per_alpha=grid)


def merge_meta_prediction_frames(
    frames: list[pd.DataFrame],
    *,
    pred_col_names: list[str],
) -> pd.DataFrame:
    if len(frames) != len(pred_col_names):
        raise ValueError("frames and pred_col_names must have the same length.")
    required = {"Season", "LowTeamID", "HighTeamID", "outcome", "pred"}
    merged = None
    for frame, pred_col in zip(frames, pred_col_names, strict=True):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Prediction frame missing required columns: {sorted(missing)}")
        current = frame[["Season", "LowTeamID", "HighTeamID", "outcome", "pred"]].rename(
            columns={"pred": pred_col}
        )
        merged = current if merged is None else merged.merge(
            current,
            on=["Season", "LowTeamID", "HighTeamID", "outcome"],
            how="inner",
            validate="one_to_one",
        )
    if merged is None:
        raise ValueError("At least one prediction frame is required.")
    return merged.sort_values(["Season", "LowTeamID", "HighTeamID"]).reset_index(drop=True)
