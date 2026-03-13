from __future__ import annotations

import pandas as pd

from mmlm2026.evaluation.men_round_group import route_group_predictions


def test_route_group_predictions_supports_women_routed_model() -> None:
    round_groups = pd.Series(["R1", "R2+", None], index=[10, 11, 12], dtype=object)
    routed = route_group_predictions(
        {
            "R1": pd.Series([0.2, 0.3, 0.4], index=[10, 11, 12], dtype=float),
            "R2+": pd.Series([0.5, 0.6, 0.7], index=[10, 11, 12], dtype=float),
        },
        round_groups,
        fallback=pd.Series([0.8, 0.85, 0.9], index=[10, 11, 12], dtype=float),
    )
    assert routed.tolist() == [0.2, 0.6, 0.9]
