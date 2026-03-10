from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmlm2026.round_utils import assign_rounds_from_seeds


def build_seed_diff_tourney_features(
    tourney_results: pd.DataFrame,
    seeds: pd.DataFrame,
    *,
    league: str,
    round_lookup_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build played-game tournament features for the seed-diff baseline.

    Output rows are oriented to the competition submission convention:
    `LowTeamID`, `HighTeamID`, and `outcome = 1` when `LowTeamID` won.

    `seed_diff` is defined as `high_seed - low_seed`, so larger positive values
    mean the lower-TeamID side has the stronger numerical seed.
    """
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

    feature_rows: list[dict[str, int | str]] = []
    for _, row in games.iterrows():
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        low_team = min(winner, loser)
        high_team = max(winner, loser)

        low_seed = seed_map[(season, low_team)]
        high_seed = seed_map[(season, high_team)]
        round_value = int(row["Round"])

        feature_rows.append(
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
        pd.DataFrame(feature_rows)
        .sort_values(["Season", "LowTeamID", "HighTeamID"])
        .reset_index(drop=True)
    )
