from __future__ import annotations

import pandas as pd

from mmlm2026.features.elo import (
    build_elo_seed_matchup_features,
    build_elo_seed_tourney_features,
    compute_end_of_regular_season_elo,
)


def test_compute_end_of_regular_season_elo_applies_day_cutoff() -> None:
    regular_season = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "DayNum": [10, 20, 140],
            "WTeamID": [10, 20, 10],
            "LTeamID": [20, 10, 20],
            "WLoc": ["N", "N", "N"],
        }
    )

    ratings = compute_end_of_regular_season_elo(regular_season, day_cutoff=134)
    rating_map = {int(row["TeamID"]): float(row["elo"]) for _, row in ratings.iterrows()}

    assert rating_map[10] < 1500.0
    assert rating_map[20] > 1500.0


def test_build_elo_seed_tourney_features_adds_elo_columns() -> None:
    results = pd.DataFrame({"Season": [2025], "WTeamID": [20], "LTeamID": [10]})
    seeds = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "Seed": ["W01", "W16"],
            "TeamID": [10, 20],
        }
    )
    elo_ratings = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "TeamID": [10, 20],
            "elo": [1600.0, 1450.0],
        }
    )

    feature_table = build_elo_seed_tourney_features(results, seeds, elo_ratings, league="M")

    assert feature_table["low_elo"].tolist() == [1600.0]
    assert feature_table["high_elo"].tolist() == [1450.0]
    assert feature_table["elo_diff"].tolist() == [150.0]
    assert feature_table["seed_diff"].tolist() == [15]


def test_build_elo_seed_matchup_features_generates_all_pairs() -> None:
    seeds = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "Seed": ["W01", "W08", "W16"],
            "TeamID": [10, 20, 30],
        }
    )
    elo_ratings = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "TeamID": [10, 20, 30],
            "elo": [1600.0, 1525.0, 1450.0],
        }
    )

    matchup_table = build_elo_seed_matchup_features(seeds, elo_ratings, season=2025, league="M")

    assert matchup_table[["LowTeamID", "HighTeamID"]].values.tolist() == [
        [10, 20],
        [10, 30],
        [20, 30],
    ]
    assert matchup_table["elo_diff"].tolist() == [75.0, 150.0, 75.0]
