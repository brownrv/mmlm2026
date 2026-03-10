from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmlm2026.round_utils import assign_rounds_from_seeds


def compute_end_of_regular_season_elo(
    regular_season_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
    initial_rating: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 100.0,
) -> pd.DataFrame:
    """Compute end-of-regular-season Elo ratings by season and team."""
    required = {"Season", "DayNum", "WTeamID", "LTeamID", "WLoc"}
    missing = required.difference(regular_season_results.columns)
    if missing:
        raise ValueError(f"Regular season results missing required columns: {sorted(missing)}")

    filtered = (
        regular_season_results.loc[regular_season_results["DayNum"] < day_cutoff]
        .sort_values(["Season", "DayNum"])
        .copy()
    )

    ratings: dict[int, dict[int, float]] = {}
    for _, row in filtered.iterrows():
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        wloc = str(row["WLoc"])

        season_ratings = ratings.setdefault(season, {})
        winner_rating = season_ratings.setdefault(winner, initial_rating)
        loser_rating = season_ratings.setdefault(loser, initial_rating)

        if wloc == "H":
            location_adjustment = home_advantage
        elif wloc == "A":
            location_adjustment = -home_advantage
        else:
            location_adjustment = 0.0

        expected_winner = 1.0 / (
            1.0 + 10 ** (((loser_rating) - (winner_rating + location_adjustment)) / 400.0)
        )
        delta = k_factor * (1.0 - expected_winner)
        season_ratings[winner] = winner_rating + delta
        season_ratings[loser] = loser_rating - delta

    rows: list[dict[str, float | int]] = []
    for season, season_ratings in ratings.items():
        for team_id, rating in season_ratings.items():
            rows.append({"Season": season, "TeamID": team_id, "elo": float(rating)})

    return pd.DataFrame(rows).sort_values(["Season", "TeamID"]).reset_index(drop=True)


def build_elo_seed_tourney_features(
    tourney_results: pd.DataFrame,
    seeds: pd.DataFrame,
    elo_ratings: pd.DataFrame,
    *,
    league: str,
    round_lookup_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build played-game tournament features for the seed-plus-Elo baseline."""
    required_elo = {"Season", "TeamID", "elo"}
    missing_elo = required_elo.difference(elo_ratings.columns)
    if missing_elo:
        raise ValueError(f"Elo ratings missing required columns: {sorted(missing_elo)}")

    seed_features = _build_seed_features_core(
        tourney_results,
        seeds,
        league=league,
        round_lookup_path=round_lookup_path,
    )
    elo_map = {
        (int(row["Season"]), int(row["TeamID"])): float(row["elo"])
        for _, row in elo_ratings.iterrows()
    }

    seed_features["low_elo"] = [
        elo_map[(int(row["Season"]), int(row["LowTeamID"]))] for _, row in seed_features.iterrows()
    ]
    seed_features["high_elo"] = [
        elo_map[(int(row["Season"]), int(row["HighTeamID"]))] for _, row in seed_features.iterrows()
    ]
    seed_features["elo_diff"] = seed_features["low_elo"] - seed_features["high_elo"]
    return seed_features


def build_elo_seed_matchup_features(
    seeds: pd.DataFrame,
    elo_ratings: pd.DataFrame,
    *,
    season: int,
    league: str,
) -> pd.DataFrame:
    """Build all seeded team-pair inference rows for a season with Elo features."""
    required_elo = {"Season", "TeamID", "elo"}
    missing_elo = required_elo.difference(elo_ratings.columns)
    if missing_elo:
        raise ValueError(f"Elo ratings missing required columns: {sorted(missing_elo)}")

    season_seeds = seeds.loc[seeds["Season"] == season].copy()
    if season_seeds.empty:
        raise ValueError(f"No seed rows found for season {season}.")

    season_seeds["seed_value"] = season_seeds["Seed"].astype(str).str[1:3].astype(int)
    season_seeds = season_seeds.sort_values("TeamID").reset_index(drop=True)
    elo_map = {
        int(row["TeamID"]): float(row["elo"])
        for _, row in elo_ratings.loc[elo_ratings["Season"] == season].iterrows()
    }

    rows: list[dict[str, float | int | str | None]] = []
    total_rows = len(season_seeds)
    for idx in range(total_rows):
        low_row = season_seeds.iloc[idx]
        for high_idx in range(idx + 1, total_rows):
            high_row = season_seeds.iloc[high_idx]
            low_team = int(low_row["TeamID"])
            high_team = int(high_row["TeamID"])
            low_elo = elo_map[low_team]
            high_elo = elo_map[high_team]
            rows.append(
                {
                    "Season": season,
                    "league": league,
                    "LowTeamID": low_team,
                    "HighTeamID": high_team,
                    "low_seed": int(low_row["seed_value"]),
                    "high_seed": int(high_row["seed_value"]),
                    "seed_diff": int(high_row["seed_value"]) - int(low_row["seed_value"]),
                    "low_elo": low_elo,
                    "high_elo": high_elo,
                    "elo_diff": low_elo - high_elo,
                    "round_group": None,
                    "Round": None,
                    "outcome": None,
                }
            )

    return pd.DataFrame(rows)


def _build_seed_features_core(
    tourney_results: pd.DataFrame,
    seeds: pd.DataFrame,
    *,
    league: str,
    round_lookup_path: str | Path | None = None,
) -> pd.DataFrame:
    required_results = {"Season", "WTeamID", "LTeamID"}
    required_seeds = {"Season", "Seed", "TeamID"}
    missing_results = required_results.difference(tourney_results.columns)
    missing_seeds = required_seeds.difference(seeds.columns)
    if missing_results:
        raise ValueError(f"Tournament results missing required columns: {sorted(missing_results)}")
    if missing_seeds:
        raise ValueError(f"Seed table missing required columns: {sorted(missing_seeds)}")

    seeds_work = seeds.copy()
    seeds_work["seed_value"] = seeds_work["Seed"].astype(str).str[1:3].astype(int)
    seed_map = {
        (int(row["Season"]), int(row["TeamID"])): int(row["seed_value"])
        for _, row in seeds_work.iterrows()
    }

    games = assign_rounds_from_seeds(
        tourney_results[["Season", "WTeamID", "LTeamID"]].copy(),
        seeds[["Season", "Seed", "TeamID"]].copy(),
        round_lookup_path=round_lookup_path,
    )
    games = games.loc[games["Round"] > 0].copy()

    rows: list[dict[str, int | str]] = []
    for _, row in games.iterrows():
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        low_team = min(winner, loser)
        high_team = max(winner, loser)
        low_seed = seed_map[(season, low_team)]
        high_seed = seed_map[(season, high_team)]
        round_value = int(row["Round"])

        rows.append(
            {
                "Season": season,
                "league": league,
                "LowTeamID": low_team,
                "HighTeamID": high_team,
                "low_seed": low_seed,
                "high_seed": high_seed,
                "seed_diff": high_seed - low_seed,
                "round_group": "R1" if round_value == 1 else "R2+",
                "Round": round_value,
                "outcome": 1 if winner == low_team else 0,
            }
        )

    return (
        pd.DataFrame(rows).sort_values(["Season", "LowTeamID", "HighTeamID"]).reset_index(drop=True)
    )
