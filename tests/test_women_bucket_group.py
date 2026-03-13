from __future__ import annotations

import pandas as pd

from mmlm2026.evaluation.women_bucket_group import (
    WomenBucketCalibrationState,
    apply_bucket_group_temperature,
    derive_focus_group,
)


def test_derive_focus_group_prioritizes_r2plus_over_bucket() -> None:
    groups = derive_focus_group(
        round_group=["R1", "R1", "R2+", "R2+"],
        bucket=["definite", "very_likely", "very_likely", "plausible"],
    )
    assert groups.tolist() == ["baseline", "very_likely", "R2+", "R2+"]


def test_apply_bucket_group_temperature_uses_fallback_for_unknown_group() -> None:
    probs = pd.Series([0.2, 0.8, 0.6], dtype=float)
    groups = pd.Series(["baseline", "very_likely", "R2+"], dtype="string")
    state = WomenBucketCalibrationState(
        temperature_by_group={"very_likely": 2.0, "R2+": 0.5},
        fallback_temperature=1.5,
    )

    adjusted = apply_bucket_group_temperature(probs, groups, state=state)

    assert len(adjusted) == 3
    assert 0.2 < adjusted.iloc[0] < 0.5
    assert 0.5 < adjusted.iloc[1] < 0.8
    assert adjusted.iloc[2] > 0.6
