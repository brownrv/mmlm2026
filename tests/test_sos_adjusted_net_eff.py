from __future__ import annotations

import pandas as pd
import pytest

from mmlm2026.features.phase_b import build_schedule_adjusted_net_eff_features
from mmlm2026.features.primary import build_phase_ab_tourney_features


def test_build_schedule_adjusted_net_eff_features_returns_expected_columns() -> None:
    efficiency = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "TeamID": [10, 20, 30],
            "off_eff": [110.0, 100.0, 95.0],
            "def_eff": [90.0, 92.0, 96.0],
        }
    )
    sos = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "TeamID": [10, 20, 30],
            "sos": [1550.0, 1500.0, 1450.0],
        }
    )

    features = build_schedule_adjusted_net_eff_features(efficiency, sos)

    assert {"Season", "TeamID", "net_eff", "sos_adjusted_net_eff"}.issubset(features.columns)
    assert features.loc[features["TeamID"] == 10, "net_eff"].iloc[0] == pytest.approx(20.0)


def test_phase_ab_tourney_features_include_sos_adjusted_net_eff_diff() -> None:
    results = pd.DataFrame({"Season": [2025], "WTeamID": [20], "LTeamID": [10]})
    seeds = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "Seed": ["W01", "W16"],
            "TeamID": [10, 20],
        }
    )
    team_features = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "TeamID": [10, 20],
            "games": [30, 30],
            "win_pct": [0.8, 0.6],
            "avg_margin": [12.0, 6.0],
            "avg_score": [78.0, 70.0],
            "avg_points_allowed": [66.0, 64.0],
            "elo": [1600.0, 1500.0],
            "tempo": [70.0, 68.0],
            "off_eff": [112.0, 104.0],
            "def_eff": [92.0, 98.0],
            "sos": [1550.0, 1490.0],
            "recent_games": [10, 10],
            "recent_win_pct": [0.8, 0.5],
            "net_eff": [20.0, 6.0],
            "sos_adjusted_net_eff": [1.5, -0.5],
        }
    )

    feature_table = build_phase_ab_tourney_features(
        results,
        seeds,
        team_features,
        league="W",
    )

    assert feature_table["sos_adjusted_net_eff_diff"].iloc[0] == pytest.approx(2.0)
