from __future__ import annotations

import pandas as pd

from mmlm2026.features.elo import (
    build_elo_seed_matchup_features,
    build_elo_seed_tourney_features,
    compute_end_of_regular_season_elo,
    compute_pre_tourney_elo_ratings,
    elo_probability_from_diff,
    pregame_expected_winner_probability,
)
from mmlm2026.features.elo_tuning import (
    EloParams,
    narrow_ranges,
    prepare_game_rows,
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


def test_compute_pre_tourney_elo_ratings_supports_carryover_and_tourney_updates() -> None:
    regular_season = pd.DataFrame(
        {
            "Season": [2024, 2025],
            "DayNum": [10, 10],
            "WTeamID": [10, 20],
            "LTeamID": [20, 10],
            "WScore": [80, 75],
            "LScore": [70, 65],
            "WLoc": ["N", "N"],
        }
    )
    tourney = pd.DataFrame(
        {
            "Season": [2024],
            "DayNum": [140],
            "WTeamID": [10],
            "LTeamID": [20],
            "WScore": [82],
            "LScore": [72],
        }
    )

    ratings = compute_pre_tourney_elo_ratings(
        regular_season,
        tourney_results=tourney,
        initial_rating=1500.0,
        k_factor=20.0,
        season_carryover=0.5,
        scale=400.0,
        mov_alpha=10.0,
        weight_regular=1.0,
        weight_tourney=1.0,
    )
    ratings_no_tourney = compute_pre_tourney_elo_ratings(
        regular_season,
        initial_rating=1500.0,
        k_factor=20.0,
        season_carryover=0.5,
        scale=400.0,
        mov_alpha=10.0,
        weight_regular=1.0,
        weight_tourney=1.0,
    )
    rating_2024 = ratings.loc[
        (ratings["Season"] == 2024) & (ratings["TeamID"] == 10),
        "elo",
    ].iloc[0]
    rating_2025 = ratings.loc[
        (ratings["Season"] == 2025) & (ratings["TeamID"] == 10),
        "elo",
    ].iloc[0]
    rating_2025_no_tourney = ratings_no_tourney.loc[
        (ratings_no_tourney["Season"] == 2025) & (ratings_no_tourney["TeamID"] == 10),
        "elo",
    ].iloc[0]

    assert rating_2024 > 1500.0
    assert rating_2025 > rating_2025_no_tourney


def test_elo_probability_from_diff_is_centered_at_half() -> None:
    probs = elo_probability_from_diff(pd.Series([0.0, 100.0, -100.0]), scale=400.0)

    assert probs.iloc[0] == 0.5
    assert probs.iloc[1] > 0.5
    assert probs.iloc[2] < 0.5


def test_pregame_expected_winner_probability_applies_loser_floor() -> None:
    probability = pregame_expected_winner_probability(
        winner_rating=100.0,
        loser_rating=-50.0,
        winner_location="N",
        scale=400.0,
        home_advantage=0.0,
    )

    assert 0.0 < probability < 1.0
    assert probability == pregame_expected_winner_probability(
        winner_rating=100.0,
        loser_rating=1.0,
        winner_location="N",
        scale=400.0,
        home_advantage=0.0,
    )


def test_prepare_game_rows_can_exclude_playins() -> None:
    regular = pd.DataFrame(
        {
            "Season": [2025],
            "DayNum": [10],
            "WTeamID": [10],
            "LTeamID": [20],
            "WScore": [70],
            "LScore": [60],
            "WLoc": ["N"],
        }
    )
    tourney = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "DayNum": [136, 138],
            "WTeamID": [10, 30],
            "LTeamID": [20, 10],
            "WScore": [68, 75],
            "LScore": [66, 70],
        }
    )
    seeds = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "Seed": ["W16a", "W16b", "W01"],
            "TeamID": [10, 20, 30],
        }
    )

    with_playins = prepare_game_rows(
        regular=regular,
        tourney=tourney,
        seeds=seeds,
        include_playins=True,
    )
    without_playins = prepare_game_rows(
        regular=regular,
        tourney=tourney,
        seeds=seeds,
        include_playins=False,
    )

    assert len(with_playins) == 3
    assert len(without_playins) == 2
    assert sum(int(row.is_tourney) for row in without_playins) == 1


def test_narrow_ranges_enforce_minimum_widths() -> None:
    best = EloParams(
        initial_rating=1000.0,
        k_factor=50.0,
        home_advantage=40.0,
        season_carryover=0.99,
        scale=300.0,
        mov_alpha=0.001,
        weight_regular=0.51,
        weight_tourney=0.5,
    )

    narrowed = narrow_ranges(
        best,
        base_ranges={
            "initial_rating": (800.0, 2000.0),
            "k_factor": (20.0, 200.0),
            "home_advantage": (30.0, 200.0),
            "season_carryover": (0.5, 1.0),
            "scale": (200.0, 2000.0),
            "mov_alpha": (0.0, 50.0),
            "weight_regular": (0.5, 1.5),
            "weight_tourney": (0.5, 1.5),
        },
    )

    mov_lower, mov_upper = narrowed["mov_alpha"]
    carry_lower, carry_upper = narrowed["season_carryover"]
    weight_lower, weight_upper = narrowed["weight_tourney"]
    assert mov_upper - mov_lower >= 1.0
    assert carry_upper - carry_lower >= 0.02
    assert weight_upper - weight_lower >= 0.1
