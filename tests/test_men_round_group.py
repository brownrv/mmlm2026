from __future__ import annotations

import pandas as pd

from mmlm2026.evaluation.men_round_group import (
    RoundGroupCalibrationState,
    apply_round_group_blend,
)


def _identity_temperature(probabilities: pd.Series, temperature: float) -> pd.Series:
    return probabilities * 0 + (probabilities.astype(float) / temperature)


def test_apply_round_group_blend_uses_group_specific_and_fallback_values() -> None:
    p_raw = pd.Series([0.6, 0.6, 0.6], index=[0, 1, 2], dtype=float)
    p_elo = pd.Series([0.4, 0.4, 0.4], index=[0, 1, 2], dtype=float)
    round_groups = pd.Series(["R1", "R2+", None], index=[0, 1, 2], dtype=object)
    state = RoundGroupCalibrationState(
        temperature_by_group={"R1": 1.0, "R2+": 2.0},
        alpha_by_group={"R1": 0.25, "R2+": 0.75},
        fallback_temperature=3.0,
        fallback_alpha=0.50,
    )

    blended = apply_round_group_blend(
        p_raw,
        p_elo,
        round_groups,
        state=state,
        apply_temperature=_identity_temperature,
    )

    assert blended.round(6).tolist() == [
        0.55,  # 0.75 * 0.6 + 0.25 * 0.4
        0.375,  # 0.25 * 0.3 + 0.75 * 0.4
        0.3,  # fallback: 0.5 * 0.2 + 0.5 * 0.4
    ]
