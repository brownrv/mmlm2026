from __future__ import annotations

import pandas as pd
import pytest

from mmlm2026.features.phase_b import (
    build_adjusted_efficiency_features,
    build_massey_consensus_features,
    build_recent_form_features,
    build_strength_of_schedule_features,
)


def test_build_adjusted_efficiency_features_returns_expected_columns() -> None:
    detailed = pd.DataFrame(
        {
            "Season": [2025],
            "DayNum": [20],
            "WTeamID": [10],
            "LTeamID": [20],
            "WScore": [80],
            "LScore": [70],
            "WFGA": [60],
            "WFTA": [20],
            "WOR": [10],
            "WTO": [12],
            "LFGA": [58],
            "LFTA": [18],
            "LOR": [8],
            "LTO": [14],
        }
    )

    features = build_adjusted_efficiency_features(detailed)

    assert set(features.columns) == {"Season", "TeamID", "games", "tempo", "off_eff", "def_eff"}
    assert set(features["TeamID"]) == {10, 20}
    assert (features["off_eff"] > 0).all()
    assert (features["def_eff"] > 0).all()


def test_build_strength_of_schedule_features_uses_opponent_elo_mean() -> None:
    regular = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "DayNum": [20, 30],
            "WTeamID": [10, 30],
            "LTeamID": [20, 10],
        }
    )
    elo = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "TeamID": [10, 20, 30],
            "elo": [1600.0, 1500.0, 1550.0],
        }
    )

    features = build_strength_of_schedule_features(regular, elo)
    sos_map = features.set_index("TeamID")["sos"].to_dict()

    assert sos_map[10] == pytest.approx((1500.0 + 1550.0) / 2)
    assert sos_map[20] == pytest.approx(1600.0)
    assert sos_map[30] == pytest.approx(1600.0)


def test_build_recent_form_features_limits_to_last_n_games() -> None:
    regular = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "DayNum": [10, 20, 30],
            "WTeamID": [10, 20, 10],
            "LTeamID": [20, 10, 30],
        }
    )

    features = build_recent_form_features(regular, last_n_games=2)
    recent_map = features.set_index("TeamID")["recent_win_pct"].to_dict()

    assert recent_map[10] == pytest.approx(0.5)
    assert recent_map[20] == pytest.approx(0.5)
    assert recent_map[30] == pytest.approx(0.0)


def test_build_massey_consensus_features_uses_latest_rank_per_system() -> None:
    massey = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025, 2025, 2025],
            "RankingDayNum": [100, 120, 110, 130, 140],
            "SystemName": ["SYS1", "SYS1", "SYS2", "SYS2", "SYS2"],
            "TeamID": [10, 10, 10, 10, 10],
            "OrdinalRank": [20, 10, 30, 25, 5],
        }
    )

    features = build_massey_consensus_features(massey, ranking_day_cutoff=133)

    assert features["massey_system_count"].iloc[0] == 2
    assert features["massey_median_rank"].iloc[0] == pytest.approx((10 + 25) / 2)
