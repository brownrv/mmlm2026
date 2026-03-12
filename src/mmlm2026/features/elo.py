from __future__ import annotations

import math
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
    return compute_pre_tourney_elo_ratings(
        regular_season_results,
        day_cutoff=day_cutoff,
        initial_rating=initial_rating,
        k_factor=k_factor,
        home_advantage=home_advantage,
    )


def compute_pre_tourney_elo_ratings(
    regular_season_results: pd.DataFrame,
    *,
    tourney_results: pd.DataFrame | None = None,
    day_cutoff: int = 134,
    initial_rating: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 100.0,
    season_carryover: float = 1.0,
    scale: float = 400.0,
    mov_alpha: float = 0.0,
    weight_regular: float = 1.0,
    weight_tourney: float = 1.0,
) -> pd.DataFrame:
    """Compute pre-tournament Elo snapshots with optional carryover and MOV weighting."""
    required = {"Season", "DayNum", "WTeamID", "LTeamID", "WLoc"}
    missing = required.difference(regular_season_results.columns)
    if missing:
        raise ValueError(f"Regular season results missing required columns: {sorted(missing)}")

    regular = (
        regular_season_results.loc[regular_season_results["DayNum"] < day_cutoff]
        .sort_values(["Season", "DayNum"])
        .copy()
    )
    if tourney_results is not None:
        required_tourney = {"Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"}
        missing_tourney = required_tourney.difference(tourney_results.columns)
        if missing_tourney:
            raise ValueError(
                f"Tournament results missing required columns: {sorted(missing_tourney)}"
            )
        tourney = tourney_results.sort_values(["Season", "DayNum"]).copy()
        tourney["WLoc"] = "N"
    else:
        tourney = None

    rows: list[dict[str, float | int]] = []
    previous_ratings: dict[int, float] = {}
    seasons = sorted(int(season) for season in regular["Season"].unique())
    for season in seasons:
        regular_games = regular.loc[regular["Season"] == season].copy()
        tourney_games = (
            tourney.loc[tourney["Season"] == season].copy() if tourney is not None else None
        )
        season_teams = set(int(team) for team in regular_games["WTeamID"].tolist())
        season_teams.update(int(team) for team in regular_games["LTeamID"].tolist())
        if tourney_games is not None and not tourney_games.empty:
            season_teams.update(int(team) for team in tourney_games["WTeamID"].tolist())
            season_teams.update(int(team) for team in tourney_games["LTeamID"].tolist())

        season_ratings = {
            team: (
                season_carryover * previous_ratings[team]
                + (1.0 - season_carryover) * initial_rating
            )
            if team in previous_ratings
            else initial_rating
            for team in sorted(season_teams)
        }

        for _, row in regular_games.iterrows():
            _apply_elo_update(
                season_ratings,
                row,
                k_factor=k_factor,
                home_advantage=home_advantage,
                scale=scale,
                mov_alpha=mov_alpha,
                weight=weight_regular,
            )

        for team_id, rating in season_ratings.items():
            rows.append({"Season": season, "TeamID": team_id, "elo": float(rating)})

        if tourney_games is not None:
            for _, row in tourney_games.iterrows():
                _apply_elo_update(
                    season_ratings,
                    row,
                    k_factor=k_factor,
                    home_advantage=0.0,
                    scale=scale,
                    mov_alpha=mov_alpha,
                    weight=weight_tourney,
                )
        previous_ratings = season_ratings

    return pd.DataFrame(rows).sort_values(["Season", "TeamID"]).reset_index(drop=True)


def elo_probability_from_diff(elo_diff: pd.Series, *, scale: float = 400.0) -> pd.Series:
    """Convert LowTeam-minus-HighTeam Elo differential into win probability."""
    scaled = (-elo_diff.astype(float) / float(scale)) * math.log(10.0)
    clipped = scaled.clip(lower=-700.0, upper=700.0)
    return 1.0 / (1.0 + clipped.map(math.exp))


def pregame_expected_winner_probability(
    winner_rating: float,
    loser_rating: float,
    *,
    winner_location: str,
    scale: float,
    home_advantage: float,
) -> float:
    """Return the pre-update, home-adjusted Elo win probability for the winner."""
    if winner_location == "H":
        location_adjustment = home_advantage
    elif winner_location == "A":
        location_adjustment = -home_advantage
    else:
        location_adjustment = 0.0

    loser_rating = max(float(loser_rating), 1.0)
    scaled = (
        (loser_rating - (float(winner_rating) + location_adjustment)) / float(scale)
    ) * math.log(10.0)
    clipped = max(min(scaled, 700.0), -700.0)
    return float(1.0 / (1.0 + math.exp(clipped)))


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


def build_elo_seed_submission_features(
    matchups: pd.DataFrame,
    seeds: pd.DataFrame,
    elo_ratings: pd.DataFrame,
    *,
    league: str,
    round_lookup_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build seed-plus-Elo features for arbitrary submission rows."""
    required_matchups = {"Season", "LowTeamID", "HighTeamID"}
    missing_matchups = required_matchups.difference(matchups.columns)
    if missing_matchups:
        raise ValueError(f"Matchup frame missing required columns: {sorted(missing_matchups)}")

    required_seeds = {"Season", "Seed", "TeamID"}
    missing_seeds = required_seeds.difference(seeds.columns)
    if missing_seeds:
        raise ValueError(f"Seed table missing required columns: {sorted(missing_seeds)}")

    required_elo = {"Season", "TeamID", "elo"}
    missing_elo = required_elo.difference(elo_ratings.columns)
    if missing_elo:
        raise ValueError(f"Elo table missing required columns: {sorted(missing_elo)}")

    enriched = matchups.copy()
    enriched["league"] = league
    enriched["outcome"] = None

    seed_values = seeds[["Season", "TeamID", "Seed"]].copy()
    seed_values["seed_value"] = seed_values["Seed"].astype(str).str[1:3].astype(int)
    enriched = (
        enriched.merge(
            seed_values.rename(
                columns={
                    "TeamID": "LowTeamID",
                    "Seed": "low_seed_label",
                    "seed_value": "low_seed",
                }
            ),
            on=["Season", "LowTeamID"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            seed_values.rename(
                columns={
                    "TeamID": "HighTeamID",
                    "Seed": "high_seed_label",
                    "seed_value": "high_seed",
                }
            ),
            on=["Season", "HighTeamID"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            elo_ratings.rename(columns={"TeamID": "LowTeamID", "elo": "low_elo"}),
            on=["Season", "LowTeamID"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            elo_ratings.rename(columns={"TeamID": "HighTeamID", "elo": "high_elo"}),
            on=["Season", "HighTeamID"],
            how="left",
            validate="many_to_one",
        )
    )

    enriched["seed_diff"] = enriched["high_seed"] - enriched["low_seed"]
    enriched["elo_diff"] = enriched["low_elo"] - enriched["high_elo"]
    rounds = _rounds_from_seed_pairs(
        enriched,
        low_seed_col="low_seed_label",
        high_seed_col="high_seed_label",
        round_lookup_path=round_lookup_path,
    )
    enriched["Round"] = rounds
    enriched["round_group"] = rounds.map(
        lambda value: (
            None
            if pd.isna(value)
            else "R0"
            if int(value) == 0
            else "R1"
            if int(value) == 1
            else "R2+"
        )
    )
    return (
        enriched.drop(columns=["low_seed_label", "high_seed_label"])
        .sort_values(["Season", "LowTeamID", "HighTeamID"])
        .reset_index(drop=True)
    )


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


def _apply_elo_update(
    ratings: dict[int, float],
    row: pd.Series,
    *,
    k_factor: float,
    home_advantage: float,
    scale: float,
    mov_alpha: float,
    weight: float,
) -> None:
    winner = int(row["WTeamID"])
    loser = int(row["LTeamID"])
    wloc = str(row.get("WLoc", "N"))
    winner_rating = ratings.setdefault(winner, 1500.0)
    loser_rating = ratings.setdefault(loser, 1500.0)

    expected_winner = pregame_expected_winner_probability(
        winner_rating,
        loser_rating,
        winner_location=wloc,
        scale=scale,
        home_advantage=home_advantage,
    )
    mov_multiplier = 1.0
    if mov_alpha > 0.0 and {"WScore", "LScore"}.issubset(row.index):
        margin = max(0.0, float(row["WScore"]) - float(row["LScore"]))
        mov_multiplier = 1.0 + margin / mov_alpha

    delta = weight * k_factor * mov_multiplier * (1.0 - expected_winner)
    ratings[winner] = winner_rating + delta
    ratings[loser] = max(loser_rating - delta, 1.0)


def _rounds_from_seed_pairs(
    frame: pd.DataFrame,
    *,
    low_seed_col: str,
    high_seed_col: str,
    round_lookup_path: str | Path | None = None,
) -> pd.Series:
    lookup_inputs = frame[["Season", low_seed_col, high_seed_col]].copy()
    required = lookup_inputs[low_seed_col].notna() & lookup_inputs[high_seed_col].notna()
    round_values = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Int64")
    if not required.any():
        return round_values

    synthetic = pd.DataFrame(
        {
            "Season": lookup_inputs.loc[required, "Season"].astype(int),
            "WTeamID": range(1, int(required.sum()) + 1),
            "LTeamID": range(10_001, 10_001 + int(required.sum())),
        }
    )
    seeds = pd.DataFrame(
        {
            "Season": pd.concat(
                [
                    lookup_inputs.loc[required, "Season"].astype(int),
                    lookup_inputs.loc[required, "Season"].astype(int),
                ],
                ignore_index=True,
            ),
            "TeamID": list(synthetic["WTeamID"].to_numpy()) + list(synthetic["LTeamID"].to_numpy()),
            "Seed": list(lookup_inputs.loc[required, low_seed_col].astype(str).to_numpy())
            + list(lookup_inputs.loc[required, high_seed_col].astype(str).to_numpy()),
        }
    )
    rounds = assign_rounds_from_seeds(
        synthetic,
        seeds,
        round_lookup_path=round_lookup_path,
    )
    round_values.loc[required] = rounds["Round"].astype("Int64").to_numpy()
    return round_values
