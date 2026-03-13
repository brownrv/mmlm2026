from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mmlm2026.round_utils import assign_rounds_from_seeds
from mmlm2026.submission.frozen_models import (
    build_seeded_submission_rows,
    load_frozen_men_context,
    load_frozen_women_context,
    predict_frozen_men,
    predict_frozen_women,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export multi-season realized-game performance for the frozen models."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/interim/frozen_historical_performance.parquet"),
    )
    parser.add_argument(
        "--season-start",
        type=int,
        default=2005,
        help="Earliest season to export. Seasons without prior training data are skipped.",
    )
    parser.add_argument(
        "--season-end",
        type=int,
        default=2025,
        help="Latest season to export.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    seasons = list(range(args.season_start, args.season_end + 1))
    men_context = load_frozen_men_context(args.data_dir)
    women_context = load_frozen_women_context(args.data_dir)

    rows: list[pd.DataFrame] = []
    skipped: list[tuple[str, int, str]] = []
    for season in seasons:
        for league in ("M", "W"):
            if season <= _minimum_trainable_season(league):
                skipped.append((league, season, "no prior training seasons"))
                continue
            try:
                context = men_context if league == "M" else women_context
                rows.append(build_historical_rows(args.data_dir, season, league, context=context))
            except ValueError as exc:
                skipped.append((league, season, str(exc)))

    if not rows:
        raise ValueError("No historical rows were generated for the requested season range.")
    performance = pd.concat(rows, ignore_index=True)
    performance = performance.sort_values(["Season", "league", "actual_round", "ID"]).reset_index(
        drop=True
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    performance.to_parquet(args.output_path, index=False)
    csv_path = args.output_path.with_suffix(".csv")
    performance.to_csv(csv_path, index=False)

    print(f"Wrote {len(performance)} rows to {args.output_path}")
    print(f"Wrote CSV companion to {csv_path}")
    if skipped:
        print("Skipped season/league combinations:")
        for league, season, reason in skipped:
            print(f"  {league} {season}: {reason}")
    return 0


def build_historical_rows(
    data_dir: Path,
    season: int,
    league: str,
    *,
    context,
) -> pd.DataFrame:
    """Build realized played-game diagnostics for one season and league."""
    config = _league_config(data_dir, league)
    seeds = pd.read_csv(config["seeds"])
    results = pd.read_csv(config["results"])

    available_seasons = set(seeds["Season"].astype(int)).intersection(results["Season"].astype(int))
    if season not in available_seasons:
        raise ValueError(f"Season {season} is not available for league {league}.")

    submission_rows = build_seeded_submission_rows(seeds, season=season)
    if league == "M":
        predictions, bracket = predict_frozen_men(
            data_dir,
            submission_rows,
            season,
            context=context,
        )
    else:
        predictions, bracket = predict_frozen_women(
            data_dir,
            submission_rows,
            season,
            context=context,
        )

    realized = assign_rounds_from_seeds(results, seeds)
    realized = realized.loc[(realized["Season"] == season) & (realized["Round"] > 0)].copy()
    realized["LowTeamID"] = realized[["WTeamID", "LTeamID"]].min(axis=1)
    realized["HighTeamID"] = realized[["WTeamID", "LTeamID"]].max(axis=1)
    realized["ID"] = realized.apply(
        lambda row: f"{int(row['Season'])}_{int(row['LowTeamID'])}_{int(row['HighTeamID'])}",
        axis=1,
    )
    realized["outcome"] = (realized["WTeamID"] == realized["LowTeamID"]).astype(int)
    realized["actual_round"] = realized["Round"].astype(int)
    realized["actual_round_group"] = realized["actual_round"].map(
        lambda value: "R1" if value == 1 else "R2+"
    )

    play_probs = bracket.play_probabilities.copy()
    play_probs["ID"] = play_probs.apply(
        lambda row: f"{int(row['Season'])}_{int(row['LowTeamID'])}_{int(row['HighTeamID'])}",
        axis=1,
    )
    play_probs["bucket"] = play_probs["play_prob"].map(bucket_from_play_prob)

    merged = (
        realized[
            [
                "Season",
                "ID",
                "LowTeamID",
                "HighTeamID",
                "outcome",
                "actual_round",
                "actual_round_group",
            ]
        ]
        .merge(
            predictions[["ID", "Pred", "Round", "round_group"]],
            on="ID",
            how="left",
            validate="one_to_one",
        )
        .merge(
            play_probs[["ID", "play_prob", "bucket"]],
            on="ID",
            how="left",
            validate="one_to_one",
        )
    )
    merged["league"] = league
    merged["predicted_round"] = merged["Round"]
    merged["predicted_round_group"] = merged["round_group"]
    merged["brier_component"] = (merged["Pred"] - merged["outcome"]).pow(2)
    return merged[
        [
            "Season",
            "league",
            "ID",
            "LowTeamID",
            "HighTeamID",
            "outcome",
            "Pred",
            "play_prob",
            "bucket",
            "actual_round",
            "actual_round_group",
            "predicted_round",
            "predicted_round_group",
            "brier_component",
        ]
    ]


def bucket_from_play_prob(play_prob: float) -> str:
    """Match the occurrence-likelihood bucket policy used by bracket diagnostics."""
    if play_prob >= 0.99:
        return "definite"
    if play_prob >= 0.30:
        return "very_likely"
    if play_prob >= 0.10:
        return "likely"
    if play_prob >= 0.03:
        return "plausible"
    return "remote"


def _league_config(data_dir: Path, league: str) -> dict[str, Path]:
    if league == "M":
        return {
            "results": data_dir / "MNCAATourneyCompactResults.csv",
            "seeds": data_dir / "MNCAATourneySeeds.csv",
        }
    if league == "W":
        return {
            "results": data_dir / "WNCAATourneyCompactResults.csv",
            "seeds": data_dir / "WNCAATourneySeeds.csv",
        }
    raise ValueError(f"Unsupported league {league!r}.")


def _minimum_trainable_season(league: str) -> int:
    if league == "M":
        return 2004
    if league == "W":
        return 2010
    raise ValueError(f"Unsupported league {league!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
