from __future__ import annotations

import pandas as pd

from mmlm2026.evaluation.men_round_group import route_group_predictions


def test_route_group_predictions_uses_group_specific_and_fallback_series() -> None:
    round_groups = pd.Series(["R1", "R2+", None], index=[0, 1, 2], dtype=object)
    routed = route_group_predictions(
        {
            "R1": pd.Series([0.1, 0.2, 0.3], index=[0, 1, 2], dtype=float),
            "R2+": pd.Series([0.4, 0.5, 0.6], index=[0, 1, 2], dtype=float),
        },
        round_groups,
        fallback=pd.Series([0.7, 0.8, 0.9], index=[0, 1, 2], dtype=float),
    )
    assert routed.tolist() == [0.1, 0.5, 0.9]
