from __future__ import annotations

import pandas as pd

from mmlm2026.evaluation.ensemble import find_best_blend_weight
from mmlm2026.evaluation.validation import build_logistic_pipeline
from mmlm2026.features.elo import elo_probability_from_diff


def select_alpha_from_prior_seasons(
    *,
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_season: int,
    elo_scale: float,
    alpha_fallback: float,
    alpha_step: float,
    alpha_calibration_seasons: int,
    train_min_games: int = 50,
) -> float:
    """Estimate a women Elo blend alpha from prior seasons only."""
    prior_seasons = sorted(
        int(season) for season in frame["Season"].unique() if season < target_season
    )
    calibration_seasons = prior_seasons[-alpha_calibration_seasons:]
    if not calibration_seasons:
        return alpha_fallback

    pooled_base: list[float] = []
    pooled_elo: list[float] = []
    pooled_outcome: list[int] = []

    for season in calibration_seasons:
        train = frame.loc[frame["Season"] < season].copy()
        valid = frame.loc[frame["Season"] == season].copy()
        if len(train) < train_min_games or valid.empty:
            continue
        model = build_logistic_pipeline(feature_cols)
        model.fit(train[feature_cols], train["outcome"].astype(int))
        base_pred = model.predict_proba(valid[feature_cols])[:, 1]
        elo_pred = elo_probability_from_diff(valid["elo_diff"], scale=elo_scale)
        pooled_base.extend(float(value) for value in base_pred)
        pooled_elo.extend(float(value) for value in elo_pred)
        pooled_outcome.extend(int(value) for value in valid["outcome"])

    if not pooled_outcome:
        return alpha_fallback

    best_alpha, _ = find_best_blend_weight(
        pooled_elo,
        pooled_base,
        pooled_outcome,
        step=alpha_step,
    )
    return float(best_alpha)
