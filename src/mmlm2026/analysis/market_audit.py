from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_betexplorer_frames(market_root: Path) -> pd.DataFrame:
    """Load BetExplorer parquet files into one canonical game table."""
    file_specs = [
        ("M", "regular", market_root / "ncaam_regular.parquet"),
        ("M", "tourney", market_root / "ncaam_tourney.parquet"),
        ("W", "regular", market_root / "ncaaw_regular.parquet"),
        ("W", "tourney", market_root / "ncaaw_tourney.parquet"),
    ]
    frames: list[pd.DataFrame] = []
    for league, stage, path in file_specs:
        if not path.exists():
            continue
        frame = pd.read_parquet(path).copy()
        frame["league"] = league
        frame["stage"] = stage
        frames.append(_canonicalize_market_frame(frame))
    if not frames:
        raise ValueError(f"No BetExplorer parquet files found under {market_root}.")
    return pd.concat(frames, ignore_index=True)


def _canonicalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame["LowTeamID"] = frame[["WTeamID", "LTeamID"]].min(axis=1)
    frame["HighTeamID"] = frame[["WTeamID", "LTeamID"]].max(axis=1)
    frame["ID"] = (
        frame["Season"].astype(int).astype(str)
        + "_"
        + frame["LowTeamID"].astype(int).astype(str)
        + "_"
        + frame["HighTeamID"].astype(int).astype(str)
    )
    winner_is_low = frame["WTeamID"] == frame["LowTeamID"]
    frame["market_pred"] = np.where(winner_is_low, frame["WProbability"], frame["LProbability"])
    frame["outcome"] = winner_is_low.astype(float)
    frame["has_market"] = frame["market_pred"].notna()
    frame["prob_sum"] = frame["WProbability"] + frame["LProbability"]
    frame["prob_sum_error"] = (frame["prob_sum"] - 1.0).abs()
    frame["prob_out_of_range"] = ~frame["market_pred"].between(0.0, 1.0, inclusive="both")
    frame.loc[~frame["has_market"], "prob_out_of_range"] = False
    return frame


def summarize_market_coverage(games: pd.DataFrame) -> pd.DataFrame:
    """Summarize market-data coverage and sanity by season, league, and stage."""
    summary = (
        games.groupby(["league", "stage", "Season"], dropna=False)
        .agg(
            games=("ID", "size"),
            games_with_market=("has_market", "sum"),
            mean_prob_sum_error=("prob_sum_error", "mean"),
            out_of_range_rows=("prob_out_of_range", "sum"),
        )
        .reset_index()
    )
    summary["coverage_rate"] = summary["games_with_market"] / summary["games"]
    summary["missing_rate"] = 1.0 - summary["coverage_rate"]
    return summary.sort_values(["league", "stage", "Season"]).reset_index(drop=True)


def compare_market_to_local_tourney(
    games: pd.DataFrame,
    local_played: pd.DataFrame,
) -> pd.DataFrame:
    """Compare market-only probabilities to frozen local predictions on played tourney rows."""
    market_tourney = games.loc[(games["stage"] == "tourney") & games["has_market"]].copy()
    local_required = {"Season", "league", "ID", "Pred", "outcome", "brier_component"}
    missing_local = local_required.difference(local_played.columns)
    if missing_local:
        raise ValueError(f"Local played-game table missing columns: {sorted(missing_local)}")
    merged = market_tourney.merge(
        local_played[list(local_required)],
        on=["Season", "league", "ID", "outcome"],
        how="inner",
        validate="one_to_one",
    )
    merged["market_brier_component"] = (merged["market_pred"] - merged["outcome"]).pow(2)
    merged["local_brier_component"] = merged["brier_component"]
    merged["market_minus_local_brier"] = (
        merged["market_brier_component"] - merged["local_brier_component"]
    )
    summary = (
        merged.groupby(["league", "Season"], dropna=False)
        .agg(
            compared_games=("ID", "size"),
            market_brier=("market_brier_component", "mean"),
            local_brier=("local_brier_component", "mean"),
            mean_market_minus_local_brier=("market_minus_local_brier", "mean"),
        )
        .reset_index()
    )
    overall = (
        merged.groupby(["league"], dropna=False)
        .agg(
            compared_games=("ID", "size"),
            market_brier=("market_brier_component", "mean"),
            local_brier=("local_brier_component", "mean"),
            mean_market_minus_local_brier=("market_minus_local_brier", "mean"),
        )
        .reset_index()
    )
    overall["Season"] = pd.NA
    return pd.concat([summary, overall], ignore_index=True)[
        [
            "league",
            "Season",
            "compared_games",
            "market_brier",
            "local_brier",
            "mean_market_minus_local_brier",
        ]
    ]


def summarize_market_viability(coverage: pd.DataFrame) -> pd.DataFrame:
    """Aggregate high-level viability signals by league and stage."""
    viability = (
        coverage.groupby(["league", "stage"], dropna=False)
        .agg(
            seasons=("Season", "nunique"),
            total_games=("games", "sum"),
            total_games_with_market=("games_with_market", "sum"),
            mean_coverage_rate=("coverage_rate", "mean"),
            median_coverage_rate=("coverage_rate", "median"),
            max_coverage_rate=("coverage_rate", "max"),
        )
        .reset_index()
    )
    viability["overall_coverage_rate"] = (
        viability["total_games_with_market"] / viability["total_games"]
    )
    return viability.sort_values(["league", "stage"]).reset_index(drop=True)
