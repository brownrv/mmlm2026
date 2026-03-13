from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RoundGroupCalibrationState:
    """Per-round-group blend parameters for the men challenger."""

    temperature_by_group: dict[str, float]
    alpha_by_group: dict[str, float]
    fallback_temperature: float
    fallback_alpha: float


def route_group_predictions(
    predictions_by_group: dict[str, pd.Series],
    round_groups: pd.Series,
    *,
    fallback: pd.Series | None = None,
) -> pd.Series:
    """Select the appropriate prediction vector for each row by round group."""
    result = pd.Series(index=round_groups.index, dtype=float)
    for group_name, predictions in predictions_by_group.items():
        mask = round_groups == group_name
        if mask.any():
            result.loc[mask] = predictions.loc[mask].astype(float)

    remaining = result.isna()
    if remaining.any():
        if fallback is None:
            raise ValueError("Round-group routing left unmatched rows without a fallback series.")
        result.loc[remaining] = fallback.loc[remaining].astype(float)
    return result


def apply_round_group_blend(
    p_raw: pd.Series,
    p_elo: pd.Series,
    round_groups: pd.Series,
    *,
    state: RoundGroupCalibrationState,
    apply_temperature: callable,
) -> pd.Series:
    """Apply round-group-specific temperature and Elo blend weights."""
    result = pd.Series(index=p_raw.index, dtype=float)
    for index, round_group in round_groups.items():
        group_key = str(round_group) if pd.notna(round_group) else ""
        temperature = state.temperature_by_group.get(group_key, state.fallback_temperature)
        alpha = state.alpha_by_group.get(group_key, state.fallback_alpha)
        p_cal = float(apply_temperature(pd.Series([float(p_raw.loc[index])]), temperature).iloc[0])
        blended = alpha * float(p_elo.loc[index]) + (1.0 - alpha) * p_cal
        result.loc[index] = min(max(blended, 1e-10), 1.0 - 1e-10)
    return result
