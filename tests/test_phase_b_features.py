from __future__ import annotations

import pandas as pd
import pytest

from mmlm2026.features.phase_b import (
    build_adjusted_efficiency_features,
    build_assist_rate_features,
    build_close_game_win_rate_features,
    build_conf_tourney_win_rate_features,
    build_free_throw_rate_features,
    build_glm_quality_features,
    build_iterative_adjusted_efficiency_features,
    build_margin_per_100_features,
    build_massey_consensus_features,
    build_massey_pca_features,
    build_recent_form_features,
    build_regularized_margin_strength_features,
    build_schedule_adjusted_net_eff_features,
    build_strength_of_schedule_features,
    build_turnover_rate_features,
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


def test_build_close_game_win_rate_features_uses_zero_imputation() -> None:
    regular = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "DayNum": [10, 20, 30],
            "WTeamID": [10, 20, 30],
            "WScore": [80, 71, 90],
            "LTeamID": [20, 10, 40],
            "LScore": [78, 70, 80],
        }
    )

    features = build_close_game_win_rate_features(regular, margin_threshold=3)
    close_map = features.set_index("TeamID")["close_win_pct"].to_dict()

    assert close_map[10] == pytest.approx(0.5)
    assert close_map[20] == pytest.approx(0.5)
    assert close_map[30] == pytest.approx(0.0)
    assert close_map[40] == pytest.approx(0.0)


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


def test_build_massey_pca_features_returns_consensus_and_disagreement() -> None:
    massey = pd.DataFrame(
        {
            "Season": [2025] * 6,
            "RankingDayNum": [120] * 6,
            "SystemName": ["SYS1", "SYS2", "SYS3", "SYS1", "SYS2", "SYS3"],
            "TeamID": [10, 10, 10, 20, 20, 20],
            "OrdinalRank": [5, 8, 6, 40, 42, 38],
        }
    )

    features = build_massey_pca_features(massey, ranking_day_cutoff=133)

    assert {"massey_pca1", "massey_disagreement"}.issubset(features.columns)
    team10 = features.loc[features["TeamID"] == 10].iloc[0]
    team20 = features.loc[features["TeamID"] == 20].iloc[0]
    assert team10["massey_pca1"] > team20["massey_pca1"]
    assert team10["massey_disagreement"] >= 0.0


def test_build_glm_quality_features_ranks_stronger_team_higher() -> None:
    regular_season = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "DayNum": [10, 20, 30],
            "WTeamID": [10, 10, 20],
            "LTeamID": [20, 30, 30],
            "WScore": [80, 78, 70],
            "LScore": [70, 60, 68],
        }
    )

    features = build_glm_quality_features(regular_season, day_cutoff=134)

    team10 = features.loc[features["TeamID"] == 10, "glm_quality"].iloc[0]
    team20 = features.loc[features["TeamID"] == 20, "glm_quality"].iloc[0]
    team30 = features.loc[features["TeamID"] == 30, "glm_quality"].iloc[0]
    assert team10 > team20 > team30
    assert features["glm_quality"].mean() == pytest.approx(0.0)


def test_build_schedule_adjusted_net_eff_features_combines_net_eff_and_sos() -> None:
    efficiency = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "TeamID": [10, 20, 30],
            "off_eff": [110.0, 102.0, 96.0],
            "def_eff": [90.0, 96.0, 98.0],
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

    assert {"net_eff", "sos_adjusted_net_eff"}.issubset(features.columns)
    assert features.loc[features["TeamID"] == 10, "net_eff"].iloc[0] == pytest.approx(20.0)


def test_men_situational_feature_builders_return_expected_values() -> None:
    detailed = pd.DataFrame(
        {
            "Season": [2025],
            "DayNum": [20],
            "WTeamID": [10],
            "LTeamID": [20],
            "WScore": [80],
            "LScore": [70],
            "WFGM": [28],
            "WFGA": [60],
            "WFTM": [12],
            "WFTA": [18],
            "WAst": [14],
            "WOR": [10],
            "WTO": [12],
            "LFGM": [24],
            "LFGA": [58],
            "LFTM": [10],
            "LFTA": [16],
            "LAst": [9],
            "LOR": [8],
            "LTO": [14],
        }
    )
    conf_tourney = pd.DataFrame(
        {
            "Season": [2025],
            "ConfAbbrev": ["sec"],
            "DayNum": [120],
            "WTeamID": [10],
            "LTeamID": [20],
        }
    )

    mov = build_margin_per_100_features(detailed)
    ast = build_assist_rate_features(detailed)
    ftr = build_free_throw_rate_features(detailed)
    tov = build_turnover_rate_features(detailed)
    conf = build_conf_tourney_win_rate_features(conf_tourney)

    assert mov.loc[mov["TeamID"] == 10, "mov_per100"].iloc[0] > 0
    assert ast.loc[ast["TeamID"] == 10, "ast_rate"].iloc[0] == pytest.approx(14 / 28)
    assert ftr.loc[ftr["TeamID"] == 10, "ftr"].iloc[0] == pytest.approx(12 / 60)
    assert tov.loc[tov["TeamID"] == 10, "tov_rate"].iloc[0] > 0
    assert conf.loc[conf["TeamID"] == 10, "conf_tourney_win_pct"].iloc[0] == pytest.approx(1.0)


def test_build_iterative_adjusted_efficiency_features_returns_adjusted_columns() -> None:
    detailed = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "DayNum": [20, 30],
            "WTeamID": [10, 20],
            "LTeamID": [20, 10],
            "WScore": [80, 70],
            "LScore": [70, 60],
            "WFGA": [60, 55],
            "WFTA": [20, 18],
            "WOR": [10, 9],
            "WTO": [12, 11],
            "LFGA": [58, 54],
            "LFTA": [18, 16],
            "LOR": [8, 7],
            "LTO": [14, 13],
        }
    )

    features = build_iterative_adjusted_efficiency_features(detailed, n_iter=20)

    assert {
        "Season",
        "TeamID",
        "adj_off_eff",
        "adj_def_eff",
        "adj_net_eff",
    }.issubset(features.columns)
    assert set(features["TeamID"]) == {10, 20}


def test_build_iterative_adjusted_efficiency_features_supports_women_hca() -> None:
    detailed = pd.DataFrame(
        {
            "Season": [2025],
            "DayNum": [20],
            "WTeamID": [10],
            "LTeamID": [20],
            "WScore": [80],
            "LScore": [70],
            "WLoc": ["H"],
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

    baseline = build_iterative_adjusted_efficiency_features(detailed, n_iter=20)
    adjusted = build_iterative_adjusted_efficiency_features(
        detailed,
        n_iter=20,
        women_hca=3.0,
    )

    baseline_team = baseline.loc[baseline["TeamID"] == 10, "raw_off_eff"].iloc[0]
    adjusted_team = adjusted.loc[adjusted["TeamID"] == 10, "raw_off_eff"].iloc[0]
    assert adjusted_team < baseline_team


def test_build_regularized_margin_strength_features_returns_centered_ratings() -> None:
    detailed = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "DayNum": [20, 30],
            "WTeamID": [10, 30],
            "LTeamID": [20, 10],
            "WScore": [80, 75],
            "LScore": [70, 65],
            "WLoc": ["H", "N"],
            "WFGA": [60, 58],
            "WFTA": [20, 18],
            "WOR": [10, 9],
            "WTO": [12, 11],
            "LFGA": [58, 56],
            "LFTA": [18, 16],
            "LOR": [8, 7],
            "LTO": [14, 13],
        }
    )

    features = build_regularized_margin_strength_features(detailed, ridge_alpha=10.0)

    assert set(features.columns) == {"Season", "TeamID", "ridge_strength"}
    assert set(features["TeamID"]) == {10, 20, 30}
    assert features["ridge_strength"].mean() == pytest.approx(0.0)
    strength_map = features.set_index("TeamID")["ridge_strength"].to_dict()
    assert strength_map[30] > strength_map[10] > strength_map[20]
