from __future__ import annotations

from pathlib import Path

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
    build_schedule_adjusted_net_eff_features,
    build_strength_of_schedule_features,
    build_turnover_rate_features,
)
from mmlm2026.round_utils import assign_rounds_from_seeds


def build_team_season_summary(
    regular_season_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
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
    return (
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
        "massey_system_count",
        "massey_median_rank",
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
    merged["win_pct_diff"] = merged["low_win_pct"] - merged["high_win_pct"]
    merged["margin_diff"] = merged["low_avg_margin"] - merged["high_avg_margin"]
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

    if {
        "low_massey_median_rank",
        "high_massey_median_rank",
    }.issubset(merged.columns):
        merged["massey_rank_diff"] = (
            merged["high_massey_median_rank"] - merged["low_massey_median_rank"]
        )

    return merged.sort_values(["Season", "LowTeamID", "HighTeamID"]).reset_index(drop=True)
