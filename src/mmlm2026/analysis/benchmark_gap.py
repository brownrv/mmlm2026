from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_reference_predictions(reference_root: Path) -> pd.DataFrame:
    """Load benchmark all-pairs predictions from partitioned parquet files."""
    rows: list[pd.DataFrame] = []
    for league_dir in sorted(reference_root.glob("League=*")):
        league = league_dir.name.split("=", maxsplit=1)[1]
        for season_dir in sorted(league_dir.glob("Season=*")):
            season = int(season_dir.name.split("=", maxsplit=1)[1])
            parquet_files = list(season_dir.glob("*.parquet"))
            if not parquet_files:
                continue
            frame = pd.read_parquet(parquet_files[0])
            frame = frame.rename(
                columns={
                    "Pred": "benchmark_pred",
                    "Round": "benchmark_round",
                    "MatchupLikelihood": "benchmark_bucket",
                    "PlayProb": "benchmark_play_prob",
                    "Occurred": "benchmark_occurred",
                    "ActualWinnerID": "benchmark_actual_winner_id",
                }
            )
            frame["league"] = league
            frame["Season"] = season
            frame["benchmark_round_group"] = frame["benchmark_round"].map(_round_group_from_round)
            rows.append(
                frame[
                    [
                        "Season",
                        "league",
                        "ID",
                        "benchmark_pred",
                        "benchmark_round",
                        "benchmark_round_group",
                        "benchmark_bucket",
                        "benchmark_play_prob",
                        "benchmark_occurred",
                        "benchmark_actual_winner_id",
                    ]
                ].copy()
            )
    if not rows:
        raise ValueError(f"No benchmark parquet files found under {reference_root}.")
    return pd.concat(rows, ignore_index=True)


def build_benchmark_gap_table(local: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Join local all-matchups predictions to benchmark outputs and derive gaps."""
    required_local = {
        "Season",
        "league",
        "ID",
        "LowTeamID",
        "HighTeamID",
        "was_played",
        "outcome",
        "Pred",
        "play_prob",
        "bucket",
        "actual_round",
        "actual_round_group",
        "brier_component",
    }
    missing_local = required_local.difference(local.columns)
    if missing_local:
        raise ValueError(f"Local all-matchups table missing columns: {sorted(missing_local)}")

    merged = local.merge(
        reference,
        on=["Season", "league", "ID"],
        how="inner",
        validate="one_to_one",
    )
    merged["pred_delta"] = merged["Pred"] - merged["benchmark_pred"]
    merged["abs_pred_delta"] = merged["pred_delta"].abs()
    merged["play_prob_delta"] = merged["play_prob"] - merged["benchmark_play_prob"]
    merged["abs_play_prob_delta"] = merged["play_prob_delta"].abs()
    merged["benchmark_brier_component"] = (merged["benchmark_pred"] - merged["outcome"]).pow(2)
    merged.loc[~merged["was_played"], "benchmark_brier_component"] = pd.NA
    merged["brier_gap"] = merged["brier_component"] - merged["benchmark_brier_component"]
    return merged


def summarize_gap_cells(merged: pd.DataFrame, *, by_season: bool = False) -> pd.DataFrame:
    """Aggregate benchmark gaps by round-group and likelihood bucket."""
    group_cols = ["league"]
    if by_season:
        group_cols.append("Season")
    group_cols.extend(["benchmark_round_group", "benchmark_bucket"])

    summary = (
        merged.groupby(group_cols, dropna=False)
        .agg(
            matchup_rows=("ID", "size"),
            played_games=("was_played", "sum"),
            seasons_present=("Season", "nunique"),
            play_prob_mass=("benchmark_play_prob", "sum"),
            mean_benchmark_play_prob=("benchmark_play_prob", "mean"),
            mean_abs_pred_delta=("abs_pred_delta", "mean"),
            mean_signed_pred_delta=("pred_delta", "mean"),
            local_brier_played=("brier_component", "mean"),
            benchmark_brier_played=("benchmark_brier_component", "mean"),
            mean_brier_gap=("brier_gap", "mean"),
        )
        .reset_index()
    )
    summary["local_brier_played"] = summary["local_brier_played"].astype(float)
    summary["benchmark_brier_played"] = summary["benchmark_brier_played"].astype(float)
    summary["mean_brier_gap"] = summary["mean_brier_gap"].astype(float)
    return summary.sort_values(group_cols).reset_index(drop=True)


def prioritize_gap_cells(summary: pd.DataFrame) -> pd.DataFrame:
    """Score recurrent benchmark gap cells by impact and recurrence."""
    eligible = summary.loc[
        (summary["benchmark_bucket"] != "not_possible")
        & (summary["played_games"] >= 8)
        & (summary["play_prob_mass"] >= 1.0)
        & (summary["mean_brier_gap"] > 0.0)
    ].copy()
    if eligible.empty:
        return eligible
    eligible["priority_score"] = (
        eligible["mean_brier_gap"] * eligible["play_prob_mass"] * eligible["played_games"]
    )
    return eligible.sort_values(
        ["priority_score", "mean_brier_gap", "play_prob_mass"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _round_group_from_round(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    value_str = str(value).strip()
    if not value_str:
        return None
    try:
        round_int = int(value_str)
    except ValueError:
        return None
    if round_int == 0:
        return "R0"
    if round_int == 1:
        return "R1"
    return "R2+"
