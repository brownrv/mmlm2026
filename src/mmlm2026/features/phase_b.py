from __future__ import annotations

import pandas as pd


def build_adjusted_efficiency_features(
    detailed_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
) -> pd.DataFrame:
    """Build season-level offensive/defensive efficiency features.

    Uses a standard possessions proxy:
    possessions = FGA - OR + TO + 0.475 * FTA
    """
    required = {
        "Season",
        "DayNum",
        "WTeamID",
        "LTeamID",
        "WScore",
        "LScore",
        "WFGA",
        "WFTA",
        "WOR",
        "WTO",
        "LFGA",
        "LFTA",
        "LOR",
        "LTO",
    }
    missing = required.difference(detailed_results.columns)
    if missing:
        raise ValueError(f"Detailed results missing required columns: {sorted(missing)}")

    filtered = detailed_results.loc[detailed_results["DayNum"] < day_cutoff].copy()
    rows: list[dict[str, float | int]] = []
    for _, row in filtered.iterrows():
        winner_poss = _estimate_possessions(
            fga=float(row["WFGA"]),
            fta=float(row["WFTA"]),
            oreb=float(row["WOR"]),
            tov=float(row["WTO"]),
        )
        loser_poss = _estimate_possessions(
            fga=float(row["LFGA"]),
            fta=float(row["LFTA"]),
            oreb=float(row["LOR"]),
            tov=float(row["LTO"]),
        )
        game_possessions = max(1.0, 0.5 * (winner_poss + loser_poss))

        rows.append(
            {
                "Season": int(row["Season"]),
                "TeamID": int(row["WTeamID"]),
                "points_for": float(row["WScore"]),
                "points_against": float(row["LScore"]),
                "game_possessions": game_possessions,
            }
        )
        rows.append(
            {
                "Season": int(row["Season"]),
                "TeamID": int(row["LTeamID"]),
                "points_for": float(row["LScore"]),
                "points_against": float(row["WScore"]),
                "game_possessions": game_possessions,
            }
        )

    season_team = pd.DataFrame(rows)
    aggregated = (
        season_team.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            games=("points_for", "size"),
            points_for=("points_for", "sum"),
            points_against=("points_against", "sum"),
            possessions=("game_possessions", "sum"),
            tempo=("game_possessions", "mean"),
        )
        .assign(
            off_eff=lambda df: 100.0 * df["points_for"] / df["possessions"],
            def_eff=lambda df: 100.0 * df["points_against"] / df["possessions"],
        )
    )
    return aggregated[["Season", "TeamID", "games", "tempo", "off_eff", "def_eff"]]


def build_strength_of_schedule_features(
    regular_season_results: pd.DataFrame,
    elo_ratings: pd.DataFrame,
    *,
    day_cutoff: int = 134,
) -> pd.DataFrame:
    """Build season-level strength-of-schedule features from opponent end-of-season Elo."""
    required_results = {"Season", "DayNum", "WTeamID", "LTeamID"}
    required_elo = {"Season", "TeamID", "elo"}
    missing_results = required_results.difference(regular_season_results.columns)
    missing_elo = required_elo.difference(elo_ratings.columns)
    if missing_results:
        raise ValueError(
            f"Regular season results missing required columns: {sorted(missing_results)}"
        )
    if missing_elo:
        raise ValueError(f"Elo ratings missing required columns: {sorted(missing_elo)}")

    filtered = regular_season_results.loc[regular_season_results["DayNum"] < day_cutoff].copy()
    rows: list[dict[str, int]] = []
    for _, row in filtered.iterrows():
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        rows.append({"Season": season, "TeamID": winner, "OpponentTeamID": loser})
        rows.append({"Season": season, "TeamID": loser, "OpponentTeamID": winner})

    games = pd.DataFrame(rows)
    elo = elo_ratings.rename(columns={"TeamID": "OpponentTeamID", "elo": "opponent_elo"})
    merged = games.merge(
        elo,
        on=["Season", "OpponentTeamID"],
        how="left",
        validate="many_to_one",
    )
    if merged["opponent_elo"].isna().any():
        raise ValueError(
            "Missing opponent Elo values while building strength-of-schedule features."
        )

    aggregated = (
        merged.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            games=("opponent_elo", "size"),
            sos=("opponent_elo", "mean"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )
    return aggregated


def build_recent_form_features(
    regular_season_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
    last_n_games: int = 10,
) -> pd.DataFrame:
    """Build season-level recent-form features from the last N pre-cutoff games."""
    required = {"Season", "DayNum", "WTeamID", "LTeamID"}
    missing = required.difference(regular_season_results.columns)
    if missing:
        raise ValueError(f"Regular season results missing required columns: {sorted(missing)}")

    filtered = regular_season_results.loc[regular_season_results["DayNum"] < day_cutoff].copy()
    rows: list[dict[str, int]] = []
    for _, row in filtered.iterrows():
        season = int(row["Season"])
        day_num = int(row["DayNum"])
        rows.append({"Season": season, "DayNum": day_num, "TeamID": int(row["WTeamID"]), "win": 1})
        rows.append({"Season": season, "DayNum": day_num, "TeamID": int(row["LTeamID"]), "win": 0})

    team_games = (
        pd.DataFrame(rows).sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)
    )
    recent = (
        team_games.groupby(["Season", "TeamID"], as_index=False)
        .tail(last_n_games)
        .groupby(["Season", "TeamID"], as_index=False)
        .agg(
            recent_games=("win", "size"),
            recent_win_pct=("win", "mean"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )
    return recent


def build_massey_consensus_features(
    massey_ordinals: pd.DataFrame,
    *,
    ranking_day_cutoff: int = 133,
) -> pd.DataFrame:
    """Build season-level median Massey rank from the last pre-cutoff ranking per system."""
    required = {"Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"}
    missing = required.difference(massey_ordinals.columns)
    if missing:
        raise ValueError(f"Massey ordinals missing required columns: {sorted(missing)}")

    filtered = massey_ordinals.loc[massey_ordinals["RankingDayNum"] <= ranking_day_cutoff].copy()
    latest_by_system = (
        filtered.sort_values(["Season", "TeamID", "SystemName", "RankingDayNum"])
        .groupby(["Season", "TeamID", "SystemName"], as_index=False)
        .tail(1)
    )
    aggregated = (
        latest_by_system.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            massey_system_count=("SystemName", "nunique"),
            massey_median_rank=("OrdinalRank", "median"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )
    return aggregated


def _estimate_possessions(*, fga: float, fta: float, oreb: float, tov: float) -> float:
    return fga - oreb + tov + 0.475 * fta
