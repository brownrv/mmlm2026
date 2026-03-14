from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mmlm2026.features.phase_b import (
    build_adjusted_efficiency_features,
    build_assist_rate_features,
    build_close_game_win_rate_features,
    build_conf_tourney_win_rate_features,
    build_free_throw_rate_features,
    build_iterative_adjusted_efficiency_features,
    build_margin_per_100_features,
    build_massey_consensus_features,
    build_recent_form_features,
    build_regularized_margin_strength_features,
    build_schedule_adjusted_net_eff_features,
    build_strength_of_schedule_features,
    build_turnover_rate_features,
)
from mmlm2026.round_utils import assign_rounds_from_seeds


def build_team_season_summary(
    regular_season_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
    pythag_exponent: float = 10.25,
) -> pd.DataFrame:
    """Build season-level win percentage and scoring summaries."""
    required = {
        "Season",
        "DayNum",
        "WTeamID",
        "WScore",
        "LTeamID",
        "LScore",
    }
    missing = required.difference(regular_season_results.columns)
    if missing:
        raise ValueError(f"Regular season results missing required columns: {sorted(missing)}")

    filtered = regular_season_results.loc[regular_season_results["DayNum"] < day_cutoff].copy()

    rows: list[dict[str, float | int]] = []
    for _, row in filtered.iterrows():
        season = int(row["Season"])
        winner_score = float(row["WScore"])
        loser_score = float(row["LScore"])
        rows.append(
            {
                "Season": season,
                "TeamID": int(row["WTeamID"]),
                "win": 1,
                "points_for": winner_score,
                "points_against": loser_score,
                "margin": winner_score - loser_score,
            }
        )
        rows.append(
            {
                "Season": season,
                "TeamID": int(row["LTeamID"]),
                "win": 0,
                "points_for": loser_score,
                "points_against": winner_score,
                "margin": loser_score - winner_score,
            }
        )

    team_games = pd.DataFrame(rows)
    summary = (
        team_games.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            games=("win", "size"),
            win_pct=("win", "mean"),
            avg_margin=("margin", "mean"),
            avg_score=("points_for", "mean"),
            avg_points_allowed=("points_against", "mean"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )
    score_power = summary["avg_score"].astype(float).clip(lower=1e-6) ** pythag_exponent
    allowed_power = summary["avg_points_allowed"].astype(float).clip(lower=1e-6) ** pythag_exponent
    summary["pythag_expectancy"] = score_power / (score_power + allowed_power)
    return summary


def build_season_momentum_features(
    regular_season_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
    split_day: int = 67,
) -> pd.DataFrame:
    """Build season-level momentum as second-half minus first-half average margin."""
    required = {
        "Season",
        "DayNum",
        "WTeamID",
        "WScore",
        "LTeamID",
        "LScore",
    }
    missing = required.difference(regular_season_results.columns)
    if missing:
        raise ValueError(f"Regular season results missing required columns: {sorted(missing)}")

    filtered = regular_season_results.loc[regular_season_results["DayNum"] < day_cutoff].copy()
    rows: list[dict[str, float | int | str]] = []
    for _, row in filtered.iterrows():
        half = "H1" if int(row["DayNum"]) < split_day else "H2"
        season = int(row["Season"])
        winner_score = float(row["WScore"])
        loser_score = float(row["LScore"])
        rows.append(
            {
                "Season": season,
                "TeamID": int(row["WTeamID"]),
                "half": half,
                "margin": winner_score - loser_score,
            }
        )
        rows.append(
            {
                "Season": season,
                "TeamID": int(row["LTeamID"]),
                "half": half,
                "margin": loser_score - winner_score,
            }
        )

    team_games = pd.DataFrame(rows)
    grouped = (
        team_games.groupby(["Season", "TeamID", "half"], as_index=False)
        .agg(avg_margin=("margin", "mean"))
        .sort_values(["Season", "TeamID", "half"])
        .reset_index(drop=True)
    )
    pivoted = grouped.pivot(index=["Season", "TeamID"], columns="half", values="avg_margin")
    pivoted = pivoted.reset_index()
    if "H1" not in pivoted.columns:
        pivoted["H1"] = 0.0
    if "H2" not in pivoted.columns:
        pivoted["H2"] = 0.0
    pivoted["H1"] = pivoted["H1"].fillna(0.0).astype(float)
    pivoted["H2"] = pivoted["H2"].fillna(0.0).astype(float)
    pivoted["season_momentum"] = pivoted["H2"] - pivoted["H1"]
    return (
        pivoted[["Season", "TeamID", "season_momentum"]]
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )


def build_late5_form_split_features(
    regular_season_detailed_results: pd.DataFrame,
    *,
    day_floor: int = 115,
    day_cutoff: int = 134,
    n_games: int = 5,
) -> pd.DataFrame:
    """Build late-season offensive and defensive form from the last N pre-tourney games."""
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
    missing = required.difference(regular_season_detailed_results.columns)
    if missing:
        raise ValueError(
            f"Regular season detailed results missing required columns: {sorted(missing)}"
        )

    filtered = regular_season_detailed_results.loc[
        (regular_season_detailed_results["DayNum"] >= day_floor)
        & (regular_season_detailed_results["DayNum"] < day_cutoff)
    ].copy()
    rows: list[dict[str, float | int]] = []
    for _, row in filtered.iterrows():
        winner_poss = (
            float(row["WFGA"]) - float(row["WOR"]) + float(row["WTO"]) + 0.475 * float(row["WFTA"])
        )
        loser_poss = (
            float(row["LFGA"]) - float(row["LOR"]) + float(row["LTO"]) + 0.475 * float(row["LFTA"])
        )
        rows.append(
            {
                "Season": int(row["Season"]),
                "DayNum": int(row["DayNum"]),
                "TeamID": int(row["WTeamID"]),
                "late5_off_eff": 100.0 * float(row["WScore"]) / max(winner_poss, 1.0),
                "late5_def_eff": 100.0 * float(row["LScore"]) / max(loser_poss, 1.0),
            }
        )
        rows.append(
            {
                "Season": int(row["Season"]),
                "DayNum": int(row["DayNum"]),
                "TeamID": int(row["LTeamID"]),
                "late5_off_eff": 100.0 * float(row["LScore"]) / max(loser_poss, 1.0),
                "late5_def_eff": 100.0 * float(row["WScore"]) / max(winner_poss, 1.0),
            }
        )

    team_games = pd.DataFrame(rows)
    if team_games.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "late5_off_eff", "late5_def_eff"])
    latest = (
        team_games.sort_values(["Season", "TeamID", "DayNum"], ascending=[True, True, False])
        .groupby(["Season", "TeamID"], group_keys=False)
        .head(n_games)
    )
    return (
        latest.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            late5_off_eff=("late5_off_eff", "mean"),
            late5_def_eff=("late5_def_eff", "mean"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )


def build_site_performance_features(
    regular_season_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
) -> pd.DataFrame:
    """Build road and neutral-site performance profiles from compact results."""
    required = {"Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"}
    missing = required.difference(regular_season_results.columns)
    if missing:
        raise ValueError(f"Regular season results missing required columns: {sorted(missing)}")

    filtered = regular_season_results.loc[regular_season_results["DayNum"] < day_cutoff].copy()
    rows: list[dict[str, float | int | str]] = []
    for _, row in filtered.iterrows():
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        winner_margin = float(row["WScore"]) - float(row["LScore"])
        loc = str(row["WLoc"])
        winner_site = "road" if loc == "A" else "neutral" if loc == "N" else "home"
        loser_site = "road" if loc == "H" else "neutral" if loc == "N" else "home"
        rows.append(
            {
                "Season": season,
                "TeamID": winner,
                "site": winner_site,
                "win": 1,
                "margin": winner_margin,
            }
        )
        rows.append(
            {
                "Season": season,
                "TeamID": loser,
                "site": loser_site,
                "win": 0,
                "margin": -winner_margin,
            }
        )

    team_games = pd.DataFrame(rows)
    if team_games.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "TeamID",
                "road_win_pct",
                "road_margin",
                "neutral_win_pct",
                "neutral_margin",
            ]
        )

    road = (
        team_games.loc[team_games["site"] == "road"]
        .groupby(["Season", "TeamID"], as_index=False)
        .agg(road_win_pct=("win", "mean"), road_margin=("margin", "mean"))
    )
    neutral = (
        team_games.loc[team_games["site"] == "neutral"]
        .groupby(["Season", "TeamID"], as_index=False)
        .agg(neutral_win_pct=("win", "mean"), neutral_margin=("margin", "mean"))
    )
    teams = team_games[["Season", "TeamID"]].drop_duplicates()
    return (
        teams.merge(road, on=["Season", "TeamID"], how="left")
        .merge(neutral, on=["Season", "TeamID"], how="left")
        .fillna(0.0)
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )


def build_win_quality_bin_features(
    regular_season_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
    close_margin: int = 5,
    blowout_margin: int = 15,
) -> pd.DataFrame:
    """Build close-game and blowout win-rate features."""
    required = {"Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"}
    missing = required.difference(regular_season_results.columns)
    if missing:
        raise ValueError(f"Regular season results missing required columns: {sorted(missing)}")

    filtered = regular_season_results.loc[regular_season_results["DayNum"] < day_cutoff].copy()
    close_rows: list[dict[str, float | int]] = []
    blowout_rows: list[dict[str, float | int]] = []
    for _, row in filtered.iterrows():
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        margin = float(row["WScore"]) - float(row["LScore"])
        if abs(margin) <= close_margin:
            close_rows.append({"Season": season, "TeamID": winner, "close_win_pct_5": 1.0})
            close_rows.append({"Season": season, "TeamID": loser, "close_win_pct_5": 0.0})
        if margin >= blowout_margin:
            blowout_rows.append({"Season": season, "TeamID": winner, "blowout_win_pct_15": 1.0})
            blowout_rows.append({"Season": season, "TeamID": loser, "blowout_win_pct_15": 0.0})

    teams = pd.concat(
        [
            filtered[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"}),
            filtered[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    close = (
        pd.DataFrame(close_rows)
        .groupby(["Season", "TeamID"], as_index=False)
        .agg(close_win_pct_5=("close_win_pct_5", "mean"))
        if close_rows
        else pd.DataFrame(columns=["Season", "TeamID", "close_win_pct_5"])
    )
    blowout = (
        pd.DataFrame(blowout_rows)
        .groupby(["Season", "TeamID"], as_index=False)
        .agg(blowout_win_pct_15=("blowout_win_pct_15", "mean"))
        if blowout_rows
        else pd.DataFrame(columns=["Season", "TeamID", "blowout_win_pct_15"])
    )
    return (
        teams.merge(close, on=["Season", "TeamID"], how="left")
        .merge(blowout, on=["Season", "TeamID"], how="left")
        .fillna(0.0)
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )


def build_conference_percentile_features(
    strength_frame: pd.DataFrame,
    team_conferences: pd.DataFrame,
    *,
    strength_col: str,
) -> pd.DataFrame:
    """Build within-conference percentile rank by a supplied strength column."""
    required_strength = {"Season", "TeamID", strength_col}
    missing_strength = required_strength.difference(strength_frame.columns)
    if missing_strength:
        raise ValueError(f"Strength frame missing required columns: {sorted(missing_strength)}")
    required_conf = {"Season", "TeamID", "ConfAbbrev"}
    missing_conf = required_conf.difference(team_conferences.columns)
    if missing_conf:
        raise ValueError(f"Team conference frame missing required columns: {sorted(missing_conf)}")

    merged = strength_frame[["Season", "TeamID", strength_col]].merge(
        team_conferences[["Season", "TeamID", "ConfAbbrev"]],
        on=["Season", "TeamID"],
        how="left",
        validate="one_to_one",
    )
    merged["conf_pct_rank"] = (
        merged.groupby(["Season", "ConfAbbrev"])[strength_col]
        .rank(method="average", pct=True)
        .astype(float)
    )
    return (
        merged[["Season", "TeamID", "conf_pct_rank"]]
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )


def build_program_pedigree_features(
    seeds: pd.DataFrame,
    tourney_results: pd.DataFrame,
    *,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """Build a simple 5-year tournament pedigree score."""
    required_seeds = {"Season", "TeamID", "Seed"}
    missing_seeds = required_seeds.difference(seeds.columns)
    if missing_seeds:
        raise ValueError(f"Seed frame missing required columns: {sorted(missing_seeds)}")
    required_results = {"Season", "WTeamID", "LTeamID"}
    missing_results = required_results.difference(tourney_results.columns)
    if missing_results:
        raise ValueError(f"Tournament results missing required columns: {sorted(missing_results)}")

    seed_work = seeds[["Season", "TeamID", "Seed"]].copy()
    seed_work["seed_value"] = (
        seed_work["Seed"].astype(str).str.extract(r"(\d+)").fillna(16).astype(int)
    )
    appearances = seed_work.groupby(["Season", "TeamID"], as_index=False).agg(
        appearances=("TeamID", "size"),
        avg_seed=("seed_value", "mean"),
        best_seed=("seed_value", "min"),
    )

    win_rows: list[dict[str, int]] = []
    for _, row in tourney_results.iterrows():
        season = int(row["Season"])
        win_rows.append({"Season": season, "TeamID": int(row["WTeamID"]), "tourney_wins": 1})
        win_rows.append({"Season": season, "TeamID": int(row["LTeamID"]), "tourney_wins": 0})
    wins = (
        pd.DataFrame(win_rows)
        .groupby(["Season", "TeamID"], as_index=False)
        .agg(tourney_wins=("tourney_wins", "sum"))
        if win_rows
        else pd.DataFrame(columns=["Season", "TeamID", "tourney_wins"])
    )

    season_team = (
        seed_work[["Season", "TeamID"]]
        .drop_duplicates()
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )
    rows: list[dict[str, float | int]] = []
    for _, row in season_team.iterrows():
        season = int(row["Season"])
        team_id = int(row["TeamID"])
        prior = appearances.loc[
            (appearances["TeamID"] == team_id)
            & (appearances["Season"] >= season - lookback_years)
            & (appearances["Season"] < season)
        ]
        prior_wins = wins.loc[
            (wins["TeamID"] == team_id)
            & (wins["Season"] >= season - lookback_years)
            & (wins["Season"] < season)
        ]
        appearances_count = float(len(prior))
        avg_seed = float(prior["avg_seed"].mean()) if not prior.empty else 16.0
        best_seed = float(prior["best_seed"].min()) if not prior.empty else 16.0
        tourney_wins = float(prior_wins["tourney_wins"].sum()) if not prior_wins.empty else 0.0
        pedigree_score = (
            appearances_count
            + 0.25 * (17.0 - avg_seed)
            + 0.5 * tourney_wins
            + 0.1 * (17.0 - best_seed)
        )
        rows.append(
            {
                "Season": season,
                "TeamID": team_id,
                "pedigree_score": pedigree_score,
            }
        )
    return pd.DataFrame(rows).sort_values(["Season", "TeamID"]).reset_index(drop=True)


def build_market_implied_strength_features(
    regular_season_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
    clip_eps: float = 1e-6,
) -> pd.DataFrame:
    """Build season-level market-implied strength summaries from pregame probabilities."""
    required = {
        "Season",
        "DayNum",
        "WTeamID",
        "LTeamID",
        "WProbability",
        "LProbability",
    }
    missing = required.difference(regular_season_results.columns)
    if missing:
        raise ValueError(f"Regular season results missing required columns: {sorted(missing)}")

    filtered = regular_season_results.loc[regular_season_results["DayNum"] < day_cutoff].copy()
    filtered = filtered.loc[
        filtered["WProbability"].notna() & filtered["LProbability"].notna()
    ].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "Season",
                "TeamID",
                "market_games",
                "market_implied_win_prob",
                "market_implied_strength",
            ]
        )

    rows: list[dict[str, float | int]] = []
    for _, row in filtered.iterrows():
        season = int(row["Season"])
        winner_prob = float(row["WProbability"])
        loser_prob = float(row["LProbability"])
        rows.append(
            {
                "Season": season,
                "TeamID": int(row["WTeamID"]),
                "market_implied_win_prob": winner_prob,
                "market_logit": np.log(
                    np.clip(winner_prob, clip_eps, 1.0 - clip_eps)
                    / np.clip(1.0 - winner_prob, clip_eps, 1.0 - clip_eps)
                ),
            }
        )
        rows.append(
            {
                "Season": season,
                "TeamID": int(row["LTeamID"]),
                "market_implied_win_prob": loser_prob,
                "market_logit": np.log(
                    np.clip(loser_prob, clip_eps, 1.0 - clip_eps)
                    / np.clip(1.0 - loser_prob, clip_eps, 1.0 - clip_eps)
                ),
            }
        )

    market_games = pd.DataFrame(rows)
    return (
        market_games.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            market_games=("market_implied_win_prob", "size"),
            market_implied_win_prob=("market_implied_win_prob", "mean"),
            market_implied_strength=("market_logit", "mean"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )


def build_phase_ab_team_features(
    regular_season_results: pd.DataFrame,
    regular_season_detailed_results: pd.DataFrame,
    elo_ratings: pd.DataFrame,
    *,
    conference_tourney_games: pd.DataFrame | None = None,
    massey_ordinals: pd.DataFrame | None = None,
    day_cutoff: int = 134,
    ranking_day_cutoff: int = 133,
    women_hca_adjustment: float | None = None,
    close_game_margin_threshold: int = 3,
) -> pd.DataFrame:
    """Build season-level Phase A+B features for each team."""
    summary = build_team_season_summary(
        regular_season_results,
        day_cutoff=day_cutoff,
    )
    efficiency = build_adjusted_efficiency_features(
        regular_season_detailed_results,
        day_cutoff=day_cutoff,
    )
    sos = build_strength_of_schedule_features(
        regular_season_results,
        elo_ratings,
        day_cutoff=day_cutoff,
    )
    recent_form = build_recent_form_features(
        regular_season_results,
        day_cutoff=day_cutoff,
    )
    close_game_form = build_close_game_win_rate_features(
        regular_season_results,
        day_cutoff=day_cutoff,
        margin_threshold=close_game_margin_threshold,
    )
    margin_per100 = build_margin_per_100_features(
        regular_season_detailed_results,
        day_cutoff=day_cutoff,
    )
    assist_rate = build_assist_rate_features(
        regular_season_detailed_results,
        day_cutoff=day_cutoff,
    )
    free_throw_rate = build_free_throw_rate_features(
        regular_season_detailed_results,
        day_cutoff=day_cutoff,
    )
    turnover_rate = build_turnover_rate_features(
        regular_season_detailed_results,
        day_cutoff=day_cutoff,
    )
    schedule_adjusted_strength = build_schedule_adjusted_net_eff_features(efficiency, sos)
    iterative_efficiency = build_iterative_adjusted_efficiency_features(
        regular_season_detailed_results,
        day_cutoff=day_cutoff,
    )
    regularized_margin_strength = build_regularized_margin_strength_features(
        regular_season_detailed_results,
        day_cutoff=day_cutoff,
    )
    women_hca_iterative_efficiency = None
    if women_hca_adjustment is not None:
        women_hca_iterative_efficiency = build_iterative_adjusted_efficiency_features(
            regular_season_detailed_results,
            day_cutoff=day_cutoff,
            women_hca=women_hca_adjustment,
        ).rename(
            columns={
                "adj_off_eff": "women_hca_adj_off_eff",
                "adj_def_eff": "women_hca_adj_def_eff",
                "adj_net_eff": "women_hca_adj_net_eff",
            }
        )

    team_features = (
        summary.merge(
            elo_ratings,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            efficiency,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            sos,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_sos"),
        )
        .merge(
            recent_form,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            close_game_form,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            margin_per100,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            assist_rate,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            free_throw_rate,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            turnover_rate,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            schedule_adjusted_strength,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            iterative_efficiency[["Season", "TeamID", "adj_off_eff", "adj_def_eff", "adj_net_eff"]],
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            regularized_margin_strength,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
    )
    if women_hca_iterative_efficiency is not None:
        team_features = team_features.merge(
            women_hca_iterative_efficiency[
                [
                    "Season",
                    "TeamID",
                    "women_hca_adj_off_eff",
                    "women_hca_adj_def_eff",
                    "women_hca_adj_net_eff",
                ]
            ],
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
    if conference_tourney_games is not None:
        conf_tourney = build_conf_tourney_win_rate_features(conference_tourney_games)
        team_features = team_features.merge(
            conf_tourney,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
        team_features["conf_tourney_games"] = (
            team_features["conf_tourney_games"].fillna(0).astype(int)
        )
        team_features["conf_tourney_win_pct"] = (
            team_features["conf_tourney_win_pct"].fillna(0.0).astype(float)
        )
    if "games_sos" in team_features.columns:
        team_features = team_features.drop(columns=["games_sos"])
    if massey_ordinals is not None:
        massey = build_massey_consensus_features(
            massey_ordinals,
            ranking_day_cutoff=ranking_day_cutoff,
        )
        team_features = team_features.merge(
            massey,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )

    return team_features.sort_values(["Season", "TeamID"]).reset_index(drop=True)


def phase_ab_feature_columns(league: str) -> list[str]:
    """Return the Phase A+B matchup feature set for a league."""
    base_cols = [
        "seed_diff",
        "elo_diff",
        "win_pct_diff",
        "margin_diff",
        "tempo_diff",
        "off_eff_diff",
        "def_eff_diff",
        "sos_diff",
        "recent_win_pct_diff",
    ]
    if league == "M":
        return [*base_cols, "massey_rank_diff"]
    return base_cols


def add_phase_c_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add matchup interaction features used by the first tree-model runs."""
    required = {
        "low_off_eff",
        "high_off_eff",
        "low_def_eff",
        "high_def_eff",
        "low_tempo",
        "high_tempo",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Matchup frame missing required Phase C source columns: {sorted(missing)}"
        )

    enriched = frame.copy()
    enriched["low_off_vs_high_def"] = enriched["low_off_eff"] - enriched["high_def_eff"]
    enriched["high_off_vs_low_def"] = enriched["high_off_eff"] - enriched["low_def_eff"]
    enriched["tempo_product"] = enriched["low_tempo"] * enriched["high_tempo"]
    return enriched


def phase_abc_feature_columns(league: str) -> list[str]:
    """Return the Phase A+B+C feature set for a league."""
    return [
        *phase_ab_feature_columns(league),
        "low_off_vs_high_def",
        "high_off_vs_low_def",
        "tempo_product",
    ]


def build_phase_ab_tourney_features(
    tourney_results: pd.DataFrame,
    seeds: pd.DataFrame,
    team_features: pd.DataFrame,
    *,
    league: str,
    round_lookup_path: str | Path | None = None,
    include_playins: bool = False,
) -> pd.DataFrame:
    """Build played-game Phase A+B matchup features in submission orientation."""
    base = _build_seed_oriented_games(
        tourney_results,
        seeds,
        league=league,
        round_lookup_path=round_lookup_path,
        include_playins=include_playins,
    )
    return _attach_team_feature_diffs(base, team_features)


def build_phase_ab_matchup_features(
    seeds: pd.DataFrame,
    team_features: pd.DataFrame,
    *,
    season: int,
    league: str,
) -> pd.DataFrame:
    """Build all seeded-team inference rows with Phase A+B features."""
    required_seeds = {"Season", "Seed", "TeamID"}
    missing_seeds = required_seeds.difference(seeds.columns)
    if missing_seeds:
        raise ValueError(f"Seed table missing required columns: {sorted(missing_seeds)}")

    season_seeds = seeds.loc[seeds["Season"] == season].copy()
    if season_seeds.empty:
        raise ValueError(f"No seed rows found for season {season}.")

    season_seeds["seed_value"] = season_seeds["Seed"].astype(str).str[1:3].astype(int)
    season_seeds = season_seeds.sort_values("TeamID").reset_index(drop=True)

    rows: list[dict[str, int | str | None]] = []
    total_rows = len(season_seeds)
    for idx in range(total_rows):
        low_row = season_seeds.iloc[idx]
        for high_idx in range(idx + 1, total_rows):
            high_row = season_seeds.iloc[high_idx]
            rows.append(
                {
                    "Season": season,
                    "league": league,
                    "LowTeamID": int(low_row["TeamID"]),
                    "HighTeamID": int(high_row["TeamID"]),
                    "low_seed": int(low_row["seed_value"]),
                    "high_seed": int(high_row["seed_value"]),
                    "seed_diff": int(high_row["seed_value"]) - int(low_row["seed_value"]),
                    "round_group": None,
                    "Round": None,
                    "outcome": None,
                }
            )

    base = pd.DataFrame(rows)
    return _attach_team_feature_diffs(base, team_features)


def build_phase_ab_submission_features(
    matchups: pd.DataFrame,
    seeds: pd.DataFrame,
    team_features: pd.DataFrame,
    *,
    league: str,
    round_lookup_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build Phase A+B features for arbitrary submission rows."""
    required_matchups = {"Season", "LowTeamID", "HighTeamID"}
    missing_matchups = required_matchups.difference(matchups.columns)
    if missing_matchups:
        raise ValueError(f"Matchup frame missing required columns: {sorted(missing_matchups)}")

    required_seeds = {"Season", "Seed", "TeamID"}
    missing_seeds = required_seeds.difference(seeds.columns)
    if missing_seeds:
        raise ValueError(f"Seed table missing required columns: {sorted(missing_seeds)}")

    base = matchups.copy()
    base["league"] = league
    base["outcome"] = None

    seed_values = seeds[["Season", "TeamID", "Seed"]].copy()
    seed_values["seed_value"] = seed_values["Seed"].astype(str).str[1:3].astype(int)
    base = base.merge(
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
    ).merge(
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
    base["seed_diff"] = base["high_seed"] - base["low_seed"]
    rounds = _rounds_from_seed_pairs(
        base,
        low_seed_col="low_seed_label",
        high_seed_col="high_seed_label",
        round_lookup_path=round_lookup_path,
    )
    base["Round"] = rounds
    base["round_group"] = rounds.map(
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
    base = base.drop(columns=["low_seed_label", "high_seed_label"])
    return _attach_team_feature_diffs(base, team_features)


def _build_seed_oriented_games(
    tourney_results: pd.DataFrame,
    seeds: pd.DataFrame,
    *,
    league: str,
    round_lookup_path: str | Path | None = None,
    include_playins: bool = False,
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

    round_inputs = tourney_results[["Season", "WTeamID", "LTeamID"]].copy()
    if {"WScore", "LScore"}.issubset(tourney_results.columns):
        round_inputs = round_inputs.join(tourney_results[["WScore", "LScore"]])

    games = assign_rounds_from_seeds(
        round_inputs,
        seeds[["Season", "Seed", "TeamID"]].copy(),
        round_lookup_path=round_lookup_path,
    )
    if include_playins:
        games = games.loc[games["Round"] >= 0].copy()
    else:
        games = games.loc[games["Round"] > 0].copy()

    rows: list[dict[str, int | str | float | None]] = []
    for _, row in games.iterrows():
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        low_team = min(winner, loser)
        high_team = max(winner, loser)
        low_seed = seed_map[(season, low_team)]
        high_seed = seed_map[(season, high_team)]
        round_value = int(row["Round"])
        margin = None
        if {"WScore", "LScore"}.issubset(tourney_results.columns):
            winner_margin = float(row["WScore"]) - float(row["LScore"])
            margin = winner_margin if winner == low_team else -winner_margin
        rows.append(
            {
                "Season": season,
                "league": league,
                "LowTeamID": low_team,
                "HighTeamID": high_team,
                "low_seed": low_seed,
                "high_seed": high_seed,
                "seed_diff": high_seed - low_seed,
                "round_group": "R0" if round_value == 0 else "R1" if round_value == 1 else "R2+",
                "Round": round_value,
                "outcome": 1 if winner == low_team else 0,
                "margin": margin,
            }
        )

    return (
        pd.DataFrame(rows).sort_values(["Season", "LowTeamID", "HighTeamID"]).reset_index(drop=True)
    )


def _attach_team_feature_diffs(
    frame: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    required = {"Season", "TeamID"}
    missing = required.difference(team_features.columns)
    if missing:
        raise ValueError(f"Team feature table missing required columns: {sorted(missing)}")

    feature_cols = [
        "games",
        "win_pct",
        "avg_margin",
        "avg_score",
        "avg_points_allowed",
        "pythag_expectancy",
        "market_games",
        "market_implied_win_prob",
        "market_implied_strength",
        "season_momentum",
        "late5_off_eff",
        "late5_def_eff",
        "road_win_pct",
        "road_margin",
        "neutral_win_pct",
        "neutral_margin",
        "close_win_pct_5",
        "blowout_win_pct_15",
        "conf_pct_rank",
        "pedigree_score",
        "elo",
        "tempo",
        "off_eff",
        "def_eff",
        "sos",
        "recent_games",
        "recent_win_pct",
        "close_games",
        "close_win_pct",
        "mov_per100",
        "ast_rate",
        "ftr",
        "tov_rate",
        "conf_tourney_games",
        "conf_tourney_win_pct",
        "net_eff",
        "sos_adjusted_net_eff",
        "adj_off_eff",
        "adj_def_eff",
        "adj_net_eff",
        "women_hca_adj_off_eff",
        "women_hca_adj_def_eff",
        "women_hca_adj_net_eff",
        "tourney_elo",
        "elo_momentum",
        "ridge_strength",
        "glm_quality",
        "espn_efg",
        "espn_tov_rate",
        "espn_orb_pct",
        "espn_ftr",
        "espn_opp_efg",
        "espn_opp_tov_rate",
        "espn_opp_orb_pct",
        "espn_opp_ftr",
        "espn_four_factor_strength",
        "espn_rotation_stability",
        "massey_system_count",
        "massey_median_rank",
        "massey_pca1",
        "massey_disagreement",
    ]
    available_feature_cols = [col for col in feature_cols if col in team_features.columns]
    features = team_features[["Season", "TeamID", *available_feature_cols]].copy()

    merged = (
        frame.merge(
            features.add_prefix("low_"),
            left_on=["Season", "LowTeamID"],
            right_on=["low_Season", "low_TeamID"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            features.add_prefix("high_"),
            left_on=["Season", "HighTeamID"],
            right_on=["high_Season", "high_TeamID"],
            how="left",
            validate="many_to_one",
        )
        .drop(columns=["low_Season", "low_TeamID", "high_Season", "high_TeamID"])
    )

    required_non_null = [
        "low_elo",
        "high_elo",
        "low_win_pct",
        "high_win_pct",
        "low_avg_margin",
        "high_avg_margin",
        "low_tempo",
        "high_tempo",
        "low_off_eff",
        "high_off_eff",
        "low_def_eff",
        "high_def_eff",
        "low_sos",
        "high_sos",
        "low_recent_win_pct",
        "high_recent_win_pct",
    ]
    missing_core = [
        col for col in required_non_null if col in merged.columns and merged[col].isna().any()
    ]
    if missing_core:
        raise ValueError(
            f"Missing core team features while building Phase A+B matchups: {sorted(missing_core)}"
        )

    merged["elo_diff"] = merged["low_elo"] - merged["high_elo"]
    if {"low_seed", "high_seed", "low_elo", "high_elo"}.issubset(merged.columns):
        low_expected_elo = 1750.0 - (merged["low_seed"].astype(float) - 1.0) * 25.0
        high_expected_elo = 1750.0 - (merged["high_seed"].astype(float) - 1.0) * 25.0
        merged["low_seed_elo_gap"] = merged["low_elo"].astype(float) - low_expected_elo
        merged["high_seed_elo_gap"] = merged["high_elo"].astype(float) - high_expected_elo
        merged["seed_elo_gap_diff"] = merged["low_seed_elo_gap"] - merged["high_seed_elo_gap"]
    merged["win_pct_diff"] = merged["low_win_pct"] - merged["high_win_pct"]
    merged["margin_diff"] = merged["low_avg_margin"] - merged["high_avg_margin"]
    if {"low_pythag_expectancy", "high_pythag_expectancy"}.issubset(merged.columns):
        merged["pythag_diff"] = merged["low_pythag_expectancy"] - merged["high_pythag_expectancy"]
    if {"low_market_implied_strength", "high_market_implied_strength"}.issubset(merged.columns):
        merged["market_implied_strength_diff"] = (
            merged["low_market_implied_strength"] - merged["high_market_implied_strength"]
        )
    if {"low_market_implied_win_prob", "high_market_implied_win_prob"}.issubset(merged.columns):
        merged["market_implied_win_prob_diff"] = (
            merged["low_market_implied_win_prob"] - merged["high_market_implied_win_prob"]
        )
    if {"low_season_momentum", "high_season_momentum"}.issubset(merged.columns):
        merged["season_momentum_diff"] = (
            merged["low_season_momentum"] - merged["high_season_momentum"]
        )
    if {"low_late5_off_eff", "high_late5_off_eff"}.issubset(merged.columns):
        merged["late5_off_diff"] = merged["low_late5_off_eff"] - merged["high_late5_off_eff"]
    if {"low_late5_def_eff", "high_late5_def_eff"}.issubset(merged.columns):
        merged["late5_def_diff"] = merged["high_late5_def_eff"] - merged["low_late5_def_eff"]
    if {"low_road_win_pct", "high_road_win_pct"}.issubset(merged.columns):
        merged["road_win_pct_diff"] = merged["low_road_win_pct"] - merged["high_road_win_pct"]
    if {"low_neutral_margin", "high_neutral_margin"}.issubset(merged.columns):
        merged["neutral_net_eff_diff"] = (
            merged["low_neutral_margin"] - merged["high_neutral_margin"]
        )
    if {"low_close_win_pct_5", "high_close_win_pct_5"}.issubset(merged.columns):
        merged["close_win_pct_5_diff"] = (
            merged["low_close_win_pct_5"] - merged["high_close_win_pct_5"]
        )
    if {"low_blowout_win_pct_15", "high_blowout_win_pct_15"}.issubset(merged.columns):
        merged["blowout_win_pct_diff"] = (
            merged["low_blowout_win_pct_15"] - merged["high_blowout_win_pct_15"]
        )
    if {"low_conf_pct_rank", "high_conf_pct_rank"}.issubset(merged.columns):
        merged["conf_pct_rank_diff"] = merged["low_conf_pct_rank"] - merged["high_conf_pct_rank"]
    if {"low_pedigree_score", "high_pedigree_score"}.issubset(merged.columns):
        merged["pedigree_score_diff"] = merged["low_pedigree_score"] - merged["high_pedigree_score"]
    merged["tempo_diff"] = merged["low_tempo"] - merged["high_tempo"]
    merged["off_eff_diff"] = merged["low_off_eff"] - merged["high_off_eff"]
    merged["def_eff_diff"] = merged["high_def_eff"] - merged["low_def_eff"]
    merged["sos_diff"] = merged["low_sos"] - merged["high_sos"]
    merged["recent_win_pct_diff"] = merged["low_recent_win_pct"] - merged["high_recent_win_pct"]
    if {"low_close_win_pct", "high_close_win_pct"}.issubset(merged.columns):
        merged["close_win_pct_diff"] = merged["low_close_win_pct"] - merged["high_close_win_pct"]
    if {"low_mov_per100", "high_mov_per100"}.issubset(merged.columns):
        merged["mov_per100_diff"] = merged["low_mov_per100"] - merged["high_mov_per100"]
    if {"low_ast_rate", "high_ast_rate"}.issubset(merged.columns):
        merged["ast_rate_diff"] = merged["low_ast_rate"] - merged["high_ast_rate"]
    if {"low_ftr", "high_ftr"}.issubset(merged.columns):
        merged["ftr_diff"] = merged["low_ftr"] - merged["high_ftr"]
    if {"low_tov_rate", "high_tov_rate"}.issubset(merged.columns):
        merged["tov_rate_diff"] = merged["low_tov_rate"] - merged["high_tov_rate"]
    if {"low_conf_tourney_win_pct", "high_conf_tourney_win_pct"}.issubset(merged.columns):
        merged["conf_tourney_win_pct_diff"] = (
            merged["low_conf_tourney_win_pct"] - merged["high_conf_tourney_win_pct"]
        )
    if {"low_net_eff", "high_net_eff"}.issubset(merged.columns):
        merged["net_eff_diff"] = merged["low_net_eff"] - merged["high_net_eff"]
    if {"low_sos_adjusted_net_eff", "high_sos_adjusted_net_eff"}.issubset(merged.columns):
        merged["sos_adjusted_net_eff_diff"] = (
            merged["low_sos_adjusted_net_eff"] - merged["high_sos_adjusted_net_eff"]
        )
    if {"low_adj_net_eff", "high_adj_net_eff"}.issubset(merged.columns):
        merged["adj_qg_diff"] = merged["low_adj_net_eff"] - merged["high_adj_net_eff"]
    if {"low_women_hca_adj_net_eff", "high_women_hca_adj_net_eff"}.issubset(merged.columns):
        merged["women_hca_adj_qg_diff"] = (
            merged["low_women_hca_adj_net_eff"] - merged["high_women_hca_adj_net_eff"]
        )
    if {"low_tourney_elo", "high_tourney_elo"}.issubset(merged.columns):
        merged["tourney_elo_diff"] = merged["low_tourney_elo"] - merged["high_tourney_elo"]
    if {"low_elo_momentum", "high_elo_momentum"}.issubset(merged.columns):
        merged["elo_momentum_diff"] = merged["low_elo_momentum"] - merged["high_elo_momentum"]
    if {"low_ridge_strength", "high_ridge_strength"}.issubset(merged.columns):
        merged["ridge_strength_diff"] = merged["low_ridge_strength"] - merged["high_ridge_strength"]
    if {"low_glm_quality", "high_glm_quality"}.issubset(merged.columns):
        merged["glm_quality_diff"] = merged["low_glm_quality"] - merged["high_glm_quality"]
    if {"low_espn_four_factor_strength", "high_espn_four_factor_strength"}.issubset(merged.columns):
        merged["espn_four_factor_strength_diff"] = (
            merged["low_espn_four_factor_strength"] - merged["high_espn_four_factor_strength"]
        )
    if {"low_espn_efg", "high_espn_efg"}.issubset(merged.columns):
        merged["espn_efg_diff"] = merged["low_espn_efg"] - merged["high_espn_efg"]
    if {"low_espn_tov_rate", "high_espn_tov_rate"}.issubset(merged.columns):
        merged["espn_tov_rate_diff"] = merged["high_espn_tov_rate"] - merged["low_espn_tov_rate"]
    if {"low_espn_orb_pct", "high_espn_orb_pct"}.issubset(merged.columns):
        merged["espn_orb_pct_diff"] = merged["low_espn_orb_pct"] - merged["high_espn_orb_pct"]
    if {"low_espn_ftr", "high_espn_ftr"}.issubset(merged.columns):
        merged["espn_ftr_diff"] = merged["low_espn_ftr"] - merged["high_espn_ftr"]
    if {"low_espn_opp_efg", "high_espn_opp_efg"}.issubset(merged.columns):
        merged["espn_opp_efg_diff"] = merged["high_espn_opp_efg"] - merged["low_espn_opp_efg"]
    if {"low_espn_opp_tov_rate", "high_espn_opp_tov_rate"}.issubset(merged.columns):
        merged["espn_opp_tov_rate_diff"] = (
            merged["low_espn_opp_tov_rate"] - merged["high_espn_opp_tov_rate"]
        )
    if {"low_espn_opp_orb_pct", "high_espn_opp_orb_pct"}.issubset(merged.columns):
        merged["espn_opp_orb_pct_diff"] = (
            merged["high_espn_opp_orb_pct"] - merged["low_espn_opp_orb_pct"]
        )
    if {"low_espn_opp_ftr", "high_espn_opp_ftr"}.issubset(merged.columns):
        merged["espn_opp_ftr_diff"] = merged["high_espn_opp_ftr"] - merged["low_espn_opp_ftr"]
    if {"low_espn_rotation_stability", "high_espn_rotation_stability"}.issubset(merged.columns):
        merged["espn_rotation_stability_diff"] = (
            merged["low_espn_rotation_stability"] - merged["high_espn_rotation_stability"]
        )

    if {
        "low_massey_median_rank",
        "high_massey_median_rank",
    }.issubset(merged.columns):
        merged["massey_rank_diff"] = (
            merged["high_massey_median_rank"] - merged["low_massey_median_rank"]
        )
    if {"low_massey_pca1", "high_massey_pca1"}.issubset(merged.columns):
        merged["massey_pca1_diff"] = merged["low_massey_pca1"] - merged["high_massey_pca1"]
    if {"low_massey_disagreement", "high_massey_disagreement"}.issubset(merged.columns):
        merged["massey_disagreement_diff"] = (
            merged["high_massey_disagreement"] - merged["low_massey_disagreement"]
        )

    return merged.sort_values(["Season", "LowTeamID", "HighTeamID"]).reset_index(drop=True)


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
