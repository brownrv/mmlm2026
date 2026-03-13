from __future__ import annotations

import pandas as pd

from mmlm2026.features.espn import (
    build_espn_men_four_factor_strength_features,
    build_espn_men_rotation_stability_features,
    build_espn_women_four_factor_strength_features,
)


def test_build_espn_men_four_factor_strength_features_matches_regular_season_games() -> None:
    boxscores = pd.DataFrame(
        {
            "gid": ["g1"] * 4 + ["g2"] * 4,
            "tid": ["100", "100", "200", "200", "100", "100", "300", "300"],
            "FG_made": [10, 12, 8, 9, 9, 8, 7, 7],
            "FG_att": [20, 22, 19, 18, 18, 17, 16, 17],
            "3PT_made": [4, 3, 2, 2, 2, 2, 1, 1],
            "FT_att": [8, 6, 7, 5, 6, 4, 4, 5],
            "OREB": [5, 4, 3, 4, 3, 3, 2, 2],
            "DREB": [10, 9, 8, 7, 9, 8, 7, 8],
            "TO": [6, 5, 8, 7, 7, 6, 9, 8],
            "PTS": [28, 27, 20, 21, 22, 20, 17, 18],
        }
    )
    regular = pd.DataFrame(
        {
            "Season": [2024],
            "WTeamID": [10],
            "LTeamID": [20],
            "WScore": [55],
            "LScore": [41],
        }
    )
    team_spellings = pd.DataFrame(
        {
            "TeamNameSpelling": ["team a", "team b", "team c"],
            "MTeamID": [10, 20, 30],
            "espn_id": ["100.0", "200.0", "300.0"],
        }
    )

    features = build_espn_men_four_factor_strength_features(
        season=2024,
        boxscores=boxscores,
        regular_season_results=regular,
        team_spellings=team_spellings,
    )

    assert set(features["TeamID"]) == {10, 20}
    assert "espn_four_factor_strength" in features.columns
    strength_map = features.set_index("TeamID")["espn_four_factor_strength"].to_dict()
    assert strength_map[10] > strength_map[20]


def test_build_espn_men_four_factor_strength_features_drops_ambiguous_mapping() -> None:
    boxscores = pd.DataFrame(
        {
            "gid": ["g1"] * 4,
            "tid": ["100", "100", "200", "200"],
            "FG_made": [10, 12, 8, 9],
            "FG_att": [20, 22, 19, 18],
            "3PT_made": [4, 3, 2, 2],
            "FT_att": [8, 6, 7, 5],
            "OREB": [5, 4, 3, 4],
            "DREB": [10, 9, 8, 7],
            "TO": [6, 5, 8, 7],
            "PTS": [28, 27, 20, 21],
        }
    )
    regular = pd.DataFrame(
        {
            "Season": [2024],
            "WTeamID": [10],
            "LTeamID": [20],
            "WScore": [55],
            "LScore": [41],
        }
    )
    team_spellings = pd.DataFrame(
        {
            "TeamNameSpelling": ["team a", "team b", "team c"],
            "MTeamID": [10, 20, 30],
            "espn_id": ["100.0", "200.0", "200.0"],
        }
    )

    features = build_espn_men_four_factor_strength_features(
        season=2024,
        boxscores=boxscores,
        regular_season_results=regular,
        team_spellings=team_spellings,
    )

    assert features.empty


def test_build_espn_women_four_factor_strength_features_matches_regular_season_games() -> None:
    boxscores = pd.DataFrame(
        {
            "gid": ["g1"] * 4 + ["g2"] * 4,
            "tid": ["100", "100", "200", "200", "100", "100", "300", "300"],
            "FG_made": [10, 12, 8, 9, 9, 8, 7, 7],
            "FG_att": [20, 22, 19, 18, 18, 17, 16, 17],
            "3PT_made": [4, 3, 2, 2, 2, 2, 1, 1],
            "FT_att": [8, 6, 7, 5, 6, 4, 4, 5],
            "OREB": [5, 4, 3, 4, 3, 3, 2, 2],
            "DREB": [10, 9, 8, 7, 9, 8, 7, 8],
            "TO": [6, 5, 8, 7, 7, 6, 9, 8],
            "PTS": [28, 27, 20, 21, 22, 20, 17, 18],
        }
    )
    regular = pd.DataFrame(
        {
            "Season": [2024],
            "WTeamID": [3101],
            "LTeamID": [3202],
            "WScore": [55],
            "LScore": [41],
        }
    )
    team_spellings = pd.DataFrame(
        {
            "TeamNameSpelling": ["team a", "team b", "team c"],
            "WTeamID": [3101, 3202, 3303],
            "espn_id": ["100.0", "200.0", "300.0"],
        }
    )

    features = build_espn_women_four_factor_strength_features(
        season=2024,
        boxscores=boxscores,
        regular_season_results=regular,
        team_spellings=team_spellings,
    )

    assert set(features["TeamID"]) == {3101, 3202}
    strength_map = features.set_index("TeamID")["espn_four_factor_strength"].to_dict()
    assert strength_map[3101] > strength_map[3202]


def test_build_espn_men_rotation_stability_features_returns_expected_columns() -> None:
    boxscores = pd.DataFrame(
        {
            "gid": ["g1"] * 6 + ["g2"] * 6,
            "tid": ["100"] * 3 + ["200"] * 3 + ["100"] * 3 + ["200"] * 3,
            "aid": ["a1", "a2", "a3", "b1", "b2", "b3", "a1", "a2", "a4", "b1", "b2", "b4"],
            "starterBench": [
                "starters",
                "starters",
                "bench",
                "starters",
                "starters",
                "bench",
                "starters",
                "starters",
                "bench",
                "starters",
                "starters",
                "bench",
            ],
            "MIN": [35, 30, 15, 36, 28, 16, 34, 31, 15, 35, 27, 18],
            "PTS": [18, 15, 6, 20, 12, 5, 17, 16, 7, 19, 13, 6],
        }
    )
    regular = pd.DataFrame(
        {
            "Season": [2024, 2024],
            "WTeamID": [10, 10],
            "LTeamID": [20, 20],
            "WScore": [39, 40],
            "LScore": [37, 38],
        }
    )
    team_spellings = pd.DataFrame(
        {
            "TeamNameSpelling": ["team a", "team b"],
            "MTeamID": [10, 20],
            "espn_id": ["100.0", "200.0"],
        }
    )

    features = build_espn_men_rotation_stability_features(
        season=2024,
        boxscores=boxscores,
        regular_season_results=regular,
        team_spellings=team_spellings,
    )

    assert {
        "Season",
        "TeamID",
        "espn_top5_minutes_share",
        "espn_top_player_minutes_share",
        "espn_starter_lineup_stability",
        "espn_rotation_stability",
    }.issubset(features.columns)
    assert set(features["TeamID"]) == {10, 20}
