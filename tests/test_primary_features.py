from __future__ import annotations

import pandas as pd
import pytest

from mmlm2026.features.primary import (
    add_phase_c_features,
    build_conference_percentile_features,
    build_late5_form_split_features,
    build_market_implied_strength_features,
    build_opponent_raw_boxscore_features,
    build_phase_ab_matchup_features,
    build_phase_ab_team_features,
    build_phase_ab_tourney_features,
    build_program_pedigree_features,
    build_season_momentum_features,
    build_site_performance_features,
    build_team_season_summary,
    build_win_quality_bin_features,
    phase_ab_feature_columns,
    phase_abc_feature_columns,
)


def test_build_team_season_summary_returns_expected_metrics() -> None:
    regular = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "DayNum": [10, 20],
            "WTeamID": [10, 20],
            "WScore": [80, 70],
            "LTeamID": [20, 10],
            "LScore": [70, 60],
        }
    )

    summary = build_team_season_summary(regular)
    win_pct_map = summary.set_index("TeamID")["win_pct"].to_dict()
    margin_map = summary.set_index("TeamID")["avg_margin"].to_dict()
    pythag_map = summary.set_index("TeamID")["pythag_expectancy"].to_dict()
    pace_map = summary.set_index("TeamID")["pace"].to_dict()

    assert win_pct_map[10] == pytest.approx(0.5)
    assert win_pct_map[20] == pytest.approx(0.5)
    assert margin_map[10] == pytest.approx(0.0)
    assert margin_map[20] == pytest.approx(0.0)
    assert 0.0 < pythag_map[10] < 1.0
    assert 0.0 < pythag_map[20] < 1.0
    assert pace_map[10] == pytest.approx(140.0)
    assert pace_map[20] == pytest.approx(140.0)


def test_build_season_momentum_features_returns_second_half_minus_first_half() -> None:
    regular = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025, 2025],
            "DayNum": [10, 20, 100, 110],
            "WTeamID": [10, 10, 20, 20],
            "WScore": [80, 78, 85, 82],
            "LTeamID": [20, 20, 10, 10],
            "LScore": [70, 74, 75, 80],
        }
    )

    momentum = build_season_momentum_features(regular)
    momentum_map = momentum.set_index("TeamID")["season_momentum"].to_dict()

    assert momentum_map[10] < 0.0
    assert momentum_map[20] > 0.0


def test_build_market_implied_strength_features_returns_team_level_summary() -> None:
    regular = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "DayNum": [10, 20],
            "WTeamID": [10, 20],
            "LTeamID": [20, 10],
            "WProbability": [0.75, 0.60],
            "LProbability": [0.25, 0.40],
        }
    )

    market = build_market_implied_strength_features(regular)
    market_map = market.set_index("TeamID")["market_implied_win_prob"].to_dict()

    assert market_map[10] == pytest.approx(0.575)
    assert market_map[20] == pytest.approx(0.425)


def test_build_opponent_raw_boxscore_features_returns_team_level_summary() -> None:
    detailed = pd.DataFrame(
        {
            "Season": [2025],
            "DayNum": [10],
            "WTeamID": [10],
            "LTeamID": [20],
            "WScore": [80],
            "LScore": [70],
            "WFGA": [60],
            "LFGA": [58],
            "WBlk": [4],
            "LBlk": [3],
            "WPF": [12],
            "LPF": [15],
            "WTO": [10],
            "LTO": [11],
            "WStl": [7],
            "LStl": [6],
        }
    )

    opp_box = build_opponent_raw_boxscore_features(detailed)
    team10 = opp_box.loc[opp_box["TeamID"] == 10].iloc[0]
    team20 = opp_box.loc[opp_box["TeamID"] == 20].iloc[0]

    assert team10["avg_opp_score"] == pytest.approx(70.0)
    assert team10["avg_opp_fga"] == pytest.approx(58.0)
    assert team10["avg_opp_blk"] == pytest.approx(3.0)
    assert team20["avg_opp_score"] == pytest.approx(80.0)
    assert team20["avg_opp_stl"] == pytest.approx(7.0)


def test_build_phase_ab_team_features_merges_phase_a_and_b_inputs() -> None:
    regular = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "DayNum": [10, 20],
            "WTeamID": [10, 20],
            "WScore": [80, 70],
            "LTeamID": [20, 10],
            "LScore": [70, 60],
        }
    )
    detailed = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "DayNum": [10, 20],
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
    elo = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "TeamID": [10, 20],
            "elo": [1600.0, 1500.0],
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
    massey = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "RankingDayNum": [120, 120],
            "SystemName": ["SYS1", "SYS1"],
            "TeamID": [10, 20],
            "OrdinalRank": [10, 30],
        }
    )

    team_features = build_phase_ab_team_features(
        regular,
        detailed,
        elo,
        conference_tourney_games=conf_tourney,
        massey_ordinals=massey,
    )

    assert {
        "win_pct",
        "avg_margin",
        "elo",
        "off_eff",
        "def_eff",
        "sos",
        "adj_off_eff",
        "adj_def_eff",
        "adj_net_eff",
        "ridge_strength",
        "close_win_pct",
        "mov_per100",
        "ast_rate",
        "ftr",
        "tov_rate",
        "conf_tourney_win_pct",
    }.issubset(team_features.columns)
    assert "massey_median_rank" in team_features.columns


def test_build_phase_ab_team_features_can_add_women_hca_adjusted_columns() -> None:
    regular = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "DayNum": [10, 20],
            "WTeamID": [10, 20],
            "WScore": [80, 70],
            "LTeamID": [20, 10],
            "LScore": [70, 60],
        }
    )
    detailed = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "DayNum": [10, 20],
            "WTeamID": [10, 20],
            "LTeamID": [20, 10],
            "WScore": [80, 70],
            "LScore": [70, 60],
            "WLoc": ["H", "N"],
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
    elo = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "TeamID": [10, 20],
            "elo": [1600.0, 1500.0],
        }
    )

    team_features = build_phase_ab_team_features(
        regular,
        detailed,
        elo,
        women_hca_adjustment=3.0,
    )

    assert {
        "women_hca_adj_off_eff",
        "women_hca_adj_def_eff",
        "women_hca_adj_net_eff",
    }.issubset(team_features.columns)


def test_build_phase_ab_tourney_and_matchup_features_compute_diffs() -> None:
    results = pd.DataFrame(
        {
            "Season": [2025],
            "WTeamID": [20],
            "LTeamID": [10],
            "WScore": [70],
            "LScore": [60],
        }
    )
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
            "pace": [144.0, 134.0],
            "avg_opp_score": [66.0, 72.0],
            "avg_opp_fga": [54.0, 58.0],
            "avg_opp_blk": [2.0, 4.0],
            "avg_opp_pf": [14.0, 17.0],
            "avg_opp_to": [11.0, 13.0],
            "avg_opp_stl": [5.0, 7.0],
            "pythag_expectancy": [0.75, 0.56],
            "season_momentum": [4.0, -2.0],
            "elo": [1600.0, 1500.0],
            "tempo": [70.0, 68.0],
            "off_eff": [112.0, 104.0],
            "def_eff": [92.0, 98.0],
            "sos": [1550.0, 1490.0],
            "adj_off_eff": [111.0, 103.0],
            "adj_def_eff": [91.0, 97.0],
            "adj_net_eff": [20.0, 6.0],
            "tourney_elo": [1525.0, 1465.0],
            "elo_momentum": [45.0, -10.0],
            "ridge_strength": [8.0, -4.0],
            "espn_efg": [0.55, 0.55],
            "espn_tov_rate": [0.14, 0.18],
            "espn_orb_pct": [0.32, 0.17],
            "espn_ftr": [0.29, 0.22],
            "espn_opp_efg": [0.42, 0.48],
            "espn_opp_tov_rate": [0.21, 0.19],
            "espn_opp_orb_pct": [0.20, 0.28],
            "espn_opp_ftr": [0.18, 0.21],
            "espn_four_factor_strength": [1.2, -0.3],
            "espn_rotation_stability": [0.8, -0.1],
            "recent_games": [10, 10],
            "recent_win_pct": [0.8, 0.5],
            "close_games": [4, 4],
            "close_win_pct": [0.75, 0.25],
            "mov_per100": [14.0, -6.0],
            "ast_rate": [0.70, 0.55],
            "ftr": [0.25, 0.18],
            "tov_rate": [0.14, 0.18],
            "conf_tourney_games": [2, 2],
            "conf_tourney_win_pct": [1.0, 0.0],
            "massey_system_count": [1, 1],
            "massey_median_rank": [10.0, 30.0],
            "massey_pca1": [1.5, -0.5],
            "massey_disagreement": [2.0, 5.0],
            "glm_quality": [4.0, -3.0],
        }
    )

    feature_table = build_phase_ab_tourney_features(
        results,
        seeds,
        team_features,
        league="M",
    )
    matchup_table = build_phase_ab_matchup_features(
        seeds,
        team_features,
        season=2025,
        league="M",
    )

    assert feature_table["seed_diff"].tolist() == [15]
    assert feature_table["elo_diff"].tolist() == [100.0]
    assert feature_table["low_seed_elo_gap"].iloc[0] == pytest.approx(-150.0)
    assert feature_table["high_seed_elo_gap"].iloc[0] == pytest.approx(125.0)
    assert feature_table["seed_elo_gap_diff"].iloc[0] == pytest.approx(-275.0)
    assert feature_table["win_pct_diff"].iloc[0] == pytest.approx(0.2)
    assert feature_table["season_momentum_diff"].iloc[0] == pytest.approx(6.0)
    assert feature_table["pace_diff"].iloc[0] == pytest.approx(10.0)
    assert feature_table["avg_opp_score_diff"].iloc[0] == pytest.approx(-6.0)
    assert feature_table["avg_opp_fga_diff"].iloc[0] == pytest.approx(-4.0)
    assert feature_table["avg_opp_blk_diff"].iloc[0] == pytest.approx(-2.0)
    assert feature_table["avg_opp_pf_diff"].iloc[0] == pytest.approx(-3.0)
    assert feature_table["avg_opp_to_diff"].iloc[0] == pytest.approx(-2.0)
    assert feature_table["avg_opp_stl_diff"].iloc[0] == pytest.approx(-2.0)
    assert feature_table["def_eff_diff"].iloc[0] == pytest.approx(6.0)
    assert feature_table["pythag_diff"].iloc[0] == pytest.approx(0.19)
    assert feature_table["massey_rank_diff"].iloc[0] == pytest.approx(20.0)
    assert feature_table["massey_pca1_diff"].iloc[0] == pytest.approx(2.0)
    assert feature_table["massey_disagreement_diff"].iloc[0] == pytest.approx(3.0)
    assert feature_table["glm_quality_diff"].iloc[0] == pytest.approx(7.0)
    assert feature_table["adj_qg_diff"].iloc[0] == pytest.approx(14.0)
    assert feature_table["tourney_elo_diff"].iloc[0] == pytest.approx(60.0)
    assert feature_table["elo_momentum_diff"].iloc[0] == pytest.approx(55.0)
    assert feature_table["ridge_strength_diff"].iloc[0] == pytest.approx(12.0)
    assert feature_table["espn_four_factor_strength_diff"].iloc[0] == pytest.approx(1.5)
    assert feature_table["espn_efg_diff"].iloc[0] == pytest.approx(0.0)
    assert feature_table["espn_tov_rate_diff"].iloc[0] == pytest.approx(0.04)
    assert feature_table["espn_orb_pct_diff"].iloc[0] == pytest.approx(0.15)
    assert feature_table["espn_ftr_diff"].iloc[0] == pytest.approx(0.07)
    assert feature_table["espn_opp_efg_diff"].iloc[0] == pytest.approx(0.06)
    assert feature_table["espn_opp_tov_rate_diff"].iloc[0] == pytest.approx(0.02)
    assert feature_table["espn_opp_orb_pct_diff"].iloc[0] == pytest.approx(0.08)
    assert feature_table["espn_opp_ftr_diff"].iloc[0] == pytest.approx(0.03)
    assert feature_table["espn_rotation_stability_diff"].iloc[0] == pytest.approx(0.9)
    assert feature_table["close_win_pct_diff"].iloc[0] == pytest.approx(0.5)
    assert feature_table["mov_per100_diff"].iloc[0] == pytest.approx(20.0)
    assert feature_table["ast_rate_diff"].iloc[0] == pytest.approx(0.15)
    assert feature_table["ftr_diff"].iloc[0] == pytest.approx(0.07)
    assert feature_table["tov_rate_diff"].iloc[0] == pytest.approx(-0.04)
    assert feature_table["conf_tourney_win_pct_diff"].iloc[0] == pytest.approx(1.0)
    assert feature_table["margin"].iloc[0] == pytest.approx(-10.0)
    assert matchup_table["elo_diff"].iloc[0] == pytest.approx(100.0)


def test_build_phase_ab_tourney_features_compute_women_hca_adj_qg_diff() -> None:
    results = pd.DataFrame(
        {
            "Season": [2025],
            "WTeamID": [20],
            "LTeamID": [10],
            "WScore": [70],
            "LScore": [60],
        }
    )
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
            "pythag_expectancy": [0.75, 0.56],
            "elo": [1600.0, 1500.0],
            "tempo": [70.0, 68.0],
            "off_eff": [112.0, 104.0],
            "def_eff": [92.0, 98.0],
            "sos": [1550.0, 1490.0],
            "adj_off_eff": [111.0, 103.0],
            "adj_def_eff": [91.0, 97.0],
            "adj_net_eff": [20.0, 6.0],
            "women_hca_adj_off_eff": [110.0, 102.0],
            "women_hca_adj_def_eff": [90.0, 98.0],
            "women_hca_adj_net_eff": [20.0, 4.0],
            "recent_games": [10, 10],
            "recent_win_pct": [0.8, 0.5],
            "close_games": [4, 4],
            "close_win_pct": [0.75, 0.25],
        }
    )

    feature_table = build_phase_ab_tourney_features(
        results,
        seeds,
        team_features,
        league="W",
    )

    assert feature_table["women_hca_adj_qg_diff"].iloc[0] == pytest.approx(16.0)


def test_phase_ab_feature_columns_excludes_massey_for_women() -> None:
    assert "massey_rank_diff" in phase_ab_feature_columns("M")
    assert "massey_rank_diff" not in phase_ab_feature_columns("W")


def test_add_phase_c_features_adds_interaction_columns() -> None:
    frame = pd.DataFrame(
        {
            "low_off_eff": [112.0],
            "high_off_eff": [104.0],
            "low_def_eff": [92.0],
            "high_def_eff": [98.0],
            "low_tempo": [70.0],
            "high_tempo": [68.0],
        }
    )

    enriched = add_phase_c_features(frame)

    assert enriched["low_off_vs_high_def"].iloc[0] == pytest.approx(14.0)
    assert enriched["high_off_vs_low_def"].iloc[0] == pytest.approx(12.0)
    assert enriched["tempo_product"].iloc[0] == pytest.approx(4760.0)
    assert "tempo_product" in phase_abc_feature_columns("W")


def test_build_late_bundle_feature_builders() -> None:
    detailed = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "DayNum": [120, 122],
            "WTeamID": [10, 20],
            "LTeamID": [20, 10],
            "WScore": [80, 71],
            "LScore": [70, 66],
            "WFGA": [60, 58],
            "WFTA": [18, 16],
            "WOR": [10, 9],
            "WTO": [12, 11],
            "LFGA": [55, 54],
            "LFTA": [15, 14],
            "LOR": [8, 7],
            "LTO": [13, 12],
        }
    )
    compact = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "DayNum": [120, 122],
            "WTeamID": [10, 20],
            "LTeamID": [20, 10],
            "WScore": [80, 71],
            "LScore": [70, 66],
            "WLoc": ["N", "A"],
        }
    )
    conferences = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "TeamID": [10, 20],
            "ConfAbbrev": ["A10", "A10"],
        }
    )
    seeds = pd.DataFrame(
        {
            "Season": [2024, 2024, 2025, 2025],
            "TeamID": [10, 20, 10, 20],
            "Seed": ["W05", "W12", "W04", "W11"],
        }
    )
    results = pd.DataFrame({"Season": [2024], "WTeamID": [10], "LTeamID": [20]})

    late5 = build_late5_form_split_features(detailed)
    site = build_site_performance_features(compact)
    quality = build_win_quality_bin_features(compact)
    summary = build_team_season_summary(compact)
    conf = build_conference_percentile_features(
        summary[["Season", "TeamID", "avg_margin"]],
        conferences,
        strength_col="avg_margin",
    )
    pedigree = build_program_pedigree_features(seeds, results)

    assert {"late5_off_eff", "late5_def_eff"}.issubset(late5.columns)
    assert {"road_win_pct", "neutral_margin"}.issubset(site.columns)
    assert {"close_win_pct_5", "blowout_win_pct_15"}.issubset(quality.columns)
    assert "conf_pct_rank" in conf.columns
    assert "pedigree_score" in pedigree.columns
