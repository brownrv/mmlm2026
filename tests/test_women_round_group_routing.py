from __future__ import annotations

import pandas as pd
from scripts.run_women_routed_round_group_model import (
    _attach_team_scalar_feature,
    _season_decay_weights,
)

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


def test_season_decay_weights_favor_recent_seasons() -> None:
    weights = _season_decay_weights(pd.Series([2021, 2023, 2024]), decay_base=0.9)

    assert weights.tolist() == [0.9**3, 0.9, 1.0]


def test_attach_team_scalar_feature_supports_defensive_sign() -> None:
    frame = pd.DataFrame({"Season": [2025], "LowTeamID": [10], "HighTeamID": [20]})
    team_feature = pd.DataFrame(
        {"Season": [2025, 2025], "TeamID": [10, 20], "late5_def_eff": [95.0, 101.0]}
    )

    enriched = _attach_team_scalar_feature(
        frame,
        team_feature,
        feature_name="late5_def_eff",
        diff_name="late5_def_diff",
        defensive=True,
    )

    assert enriched.loc[0, "late5_def_diff"] == 6.0
