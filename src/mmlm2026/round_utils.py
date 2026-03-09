"""Canonical round assignment for NCAA tournament games.

Uses seed-pair lookup via data/tourney_round_lookup.csv.  Works identically
for men and women across all seasons, including the 2021 men's bubble schedule
and era-varying women's DayNums.

Approach
--------
1. Map each game's WTeamID / LTeamID to their Seed via NCAATourneySeeds.
2. Normalize seeds by stripping trailing 'a'/'b' (First-Four designations).
3. Look up (seed1_norm, seed2_norm) in tourney_round_lookup → Round.
4. Play-in games have identical normalized seeds (e.g. W16a vs W16b → W16 == W16) → Round 0.

DO NOT use DayNum for round assignment.  DayNum < 134 filtering for Elo
game cutoffs is unrelated and remains correct.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DEFAULT_LOOKUP = Path(__file__).parent.parent.parent / "data" / "tourney_round_lookup.csv"


def assign_rounds_from_seeds(
    tc: pd.DataFrame,
    seeds: pd.DataFrame,
    round_lookup_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return *tc* with a new integer 'Round' column (0=PlayIn … 6=Champ).

    Parameters
    ----------
    tc : DataFrame
        Tournament compact results with columns Season, WTeamID, LTeamID.
    seeds : DataFrame
        NCAATourneySeeds with columns Season, TeamID, Seed.
    round_lookup_path : path-like, optional
        Path to tourney_round_lookup.csv.  Defaults to data/tourney_round_lookup.csv
        relative to the repository root.

    Returns
    -------
    DataFrame
        Copy of *tc* with 'Round' column added (int, -1 if unresolvable).
    """
    path = Path(round_lookup_path) if round_lookup_path is not None else _DEFAULT_LOOKUP
    rl = pd.read_csv(path)

    # Build bidirectional (seed_norm, seed_norm) -> Round lookup
    rl_dict: dict[tuple[str, str], int] = {}
    for _, row in rl.iterrows():
        s = str(row["StrongSeed"]).rstrip("ab")
        w = str(row["WeakSeed"]).rstrip("ab")
        rnd = int(row["Round"])
        rl_dict[(s, w)] = rnd
        rl_dict[(w, s)] = rnd

    seed_map: dict[tuple[int, int], str] = {
        (int(r["Season"]), int(r["TeamID"])): str(r["Seed"]) for _, r in seeds.iterrows()
    }

    tc = tc.copy()
    ws = tc.apply(lambda r: seed_map.get((int(r["Season"]), int(r["WTeamID"])), ""), axis=1)
    ls = tc.apply(lambda r: seed_map.get((int(r["Season"]), int(r["LTeamID"])), ""), axis=1)
    ws_norm = ws.str.rstrip("ab")
    ls_norm = ls.str.rstrip("ab")

    is_playin = ws_norm == ls_norm
    tc["Round"] = [
        rl_dict.get((winner_seed, loser_seed), -1)
        for winner_seed, loser_seed in zip(ws_norm, ls_norm, strict=False)
    ]
    tc.loc[is_playin, "Round"] = 0
    return tc
