from __future__ import annotations

import pandas as pd

from mmlm2026.features.baseline import build_seed_diff_tourney_features


def test_build_seed_diff_tourney_features_orients_rows_to_low_team() -> None:
    results = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "WTeamID": [20, 40],
            "LTeamID": [10, 30],
        }
    )
    seeds = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025, 2025],
            "Seed": ["W01", "W16", "X05", "X12"],
            "TeamID": [10, 20, 30, 40],
        }
    )

    feature_table = build_seed_diff_tourney_features(results, seeds, league="M")

    assert feature_table["LowTeamID"].tolist() == [10, 30]
    assert feature_table["HighTeamID"].tolist() == [20, 40]
    assert feature_table["outcome"].tolist() == [0, 0]
    assert feature_table["seed_diff"].tolist() == [15, 7]
    assert set(feature_table["round_group"]) == {"R1"}
