from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss  # type: ignore[import-untyped]


@dataclass(frozen=True)
class WomenBucketCalibrationState:
    """Temperature state for targeted women bucket and round-group adjustments."""

    temperature_by_group: dict[str, float]
    fallback_temperature: float


def derive_focus_group(
    round_group: pd.Series | list[str],
    bucket: pd.Series | list[str],
) -> pd.Series:
    """Map rows to the narrow women challenger groups."""
    round_series = pd.Series(round_group, dtype="string")
    bucket_series = pd.Series(bucket, dtype="string")
    labels = pd.Series("baseline", index=round_series.index, dtype="string")
    labels.loc[bucket_series == "very_likely"] = "very_likely"
    labels.loc[round_series == "R2+"] = "R2+"
    return labels


def apply_bucket_group_temperature(
    probabilities: pd.Series | list[float],
    focus_group: pd.Series | list[str],
    *,
    state: WomenBucketCalibrationState,
) -> pd.Series:
    """Apply group-specific temperature scaling with a global fallback."""
    prob_series = pd.Series(probabilities, dtype=float)
    group_series = pd.Series(focus_group, index=prob_series.index, dtype="string")
    adjusted = pd.Series(index=prob_series.index, dtype=float)
    for group_name, group_temp in state.temperature_by_group.items():
        mask = group_series == group_name
        if mask.any():
            adjusted.loc[mask] = apply_temperature(prob_series.loc[mask], group_temp)

    remaining = adjusted.isna()
    if remaining.any():
        adjusted.loc[remaining] = apply_temperature(
            prob_series.loc[remaining],
            state.fallback_temperature,
        )
    return adjusted.astype(float)


def apply_temperature(
    probabilities: pd.Series | list[float],
    temperature: float,
) -> pd.Series:
    """Apply logistic temperature scaling to probabilities."""
    prob_series = pd.Series(probabilities, dtype=float)
    logits = logit(prob_series) / float(temperature)
    calibrated = 1.0 / (1.0 + np.exp(-logits))
    return pd.Series(calibrated, index=prob_series.index, dtype=float).clip(
        lower=1e-10,
        upper=1.0 - 1e-10,
    )


def optimize_temperature(
    probabilities: pd.Series | list[float],
    outcomes: pd.Series | list[int],
) -> float:
    """Select the temperature that minimizes log-loss on the supplied rows."""
    prob_series = pd.Series(probabilities, dtype=float)
    outcome_series = pd.Series(outcomes, dtype=int)
    grid = np.linspace(0.1, 5.0, 200)
    return float(
        min(
            grid,
            key=lambda temperature: float(
                log_loss(
                    outcome_series,
                    apply_temperature(prob_series, float(temperature)),
                )
            ),
        )
    )


def logit(probabilities: pd.Series | list[float]) -> pd.Series:
    """Compute logits with clipping to keep the transform stable."""
    prob_series = pd.Series(probabilities, dtype=float).clip(1e-10, 1.0 - 1e-10)
    logits = np.log(prob_series / (1.0 - prob_series))
    return pd.Series(logits, index=prob_series.index, dtype=float)
