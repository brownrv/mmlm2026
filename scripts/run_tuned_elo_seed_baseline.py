from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts
from mmlm2026.evaluation.validation import (
    build_logistic_pipeline,
    save_validation_artifacts,
    validate_season_holdouts,
)
from mmlm2026.features.elo import (
    build_elo_seed_matchup_features,
    build_elo_seed_tourney_features,
    compute_pre_tourney_elo_ratings,
)
from mmlm2026.utils.mlflow_tracking import start_tracked_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the tuned seed-plus-Elo tournament baseline.")
    parser.add_argument("--league", choices=["M", "W"], required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument("--holdout-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/baseline_runs"))
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--day-cutoff", type=int, default=134)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    regular_season_path, tourney_results_path, seeds_path, slots_path = _league_paths(
        args.data_dir,
        args.league,
    )
    regular_season = pd.read_csv(regular_season_path)
    results = pd.read_csv(tourney_results_path)
    seeds = pd.read_csv(seeds_path)
    slots = pd.read_csv(slots_path)

    elo_kwargs = _elo_params(args.league)
    elo_ratings = compute_pre_tourney_elo_ratings(
        regular_season,
        tourney_results=results,
        day_cutoff=args.day_cutoff,
        **elo_kwargs,
    )
    feature_table = build_elo_seed_tourney_features(results, seeds, elo_ratings, league=args.league)

    output_dir = args.output_dir / f"{args.league.lower()}_elo_tuned"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_table.to_parquet(output_dir / "elo_seed_features.parquet", index=False)
        elo_ratings.to_parquet(output_dir / "elo_ratings.parquet", index=False)

    validation = validate_season_holdouts(
        feature_table,
        feature_cols=["seed_diff", "elo_diff"],
        holdout_seasons=args.holdout_seasons,
        train_min_games=50,
    )
    validation_artifacts = save_validation_artifacts(
        validation,
        output_dir=output_dir / "validation",
    )

    latest_holdout = max(args.holdout_seasons)
    train_frame = feature_table.loc[feature_table["Season"] < latest_holdout].copy()
    infer_frame = build_elo_seed_matchup_features(
        seeds,
        elo_ratings,
        season=latest_holdout,
        league=args.league,
    )
    model = build_logistic_pipeline(["seed_diff", "elo_diff"])
    model.fit(train_frame[["seed_diff", "elo_diff"]], train_frame["outcome"].astype(int))
    infer_frame["Pred"] = model.predict_proba(infer_frame[["seed_diff", "elo_diff"]])[:, 1]
    infer_frame["ID"] = [
        f"{latest_holdout}_{low}_{high}"
        for low, high in zip(
            infer_frame["LowTeamID"],
            infer_frame["HighTeamID"],
            strict=False,
        )
    ]

    bracket = compute_bracket_diagnostics(
        slots,
        seeds,
        infer_frame[["ID", "Pred"]],
        season=latest_holdout,
        results=results,
        round_col="Round",
    )
    bracket_artifacts = save_bracket_artifacts(bracket, output_dir=output_dir / "bracket")

    if args.log_mlflow:
        _log_mlflow_run(args, validation, validation_artifacts, bracket_artifacts, elo_kwargs)

    print(validation.per_season_metrics.to_string(index=False))
    print(
        "\nOverall metrics:",
        f"flat_brier={validation.overall_flat_brier:.6f}",
        f"log_loss={validation.overall_log_loss:.6f}",
    )
    print(f"Artifacts written under {output_dir}")
    return 0


def _league_paths(data_dir: Path, league: str) -> tuple[Path, Path, Path, Path]:
    prefix = "M" if league == "M" else "W"
    regular_season_path = data_dir / f"{prefix}RegularSeasonCompactResults.csv"
    tourney_results_path = data_dir / f"{prefix}NCAATourneyCompactResults.csv"
    seeds_path = data_dir / f"{prefix}NCAATourneySeeds.csv"
    slots_path = data_dir / f"{prefix}NCAATourneySlots.csv"
    return regular_season_path, tourney_results_path, seeds_path, slots_path


def _elo_params(league: str) -> dict[str, float]:
    if league == "M":
        return {
            "initial_rating": 1618.0,
            "k_factor": 76.0,
            "home_advantage": 43.0,
            "season_carryover": 0.9780,
            "scale": 1835.34,
            "mov_alpha": 6.5450,
            "weight_regular": 1.4674,
            "weight_tourney": 0.8204,
        }
    return {
        "initial_rating": 1213.0,
        "k_factor": 136.0,
        "home_advantage": 119.0,
        "season_carryover": 0.9840,
        "scale": 1888.27,
        "mov_alpha": 11.4612,
        "weight_regular": 1.4974,
        "weight_tourney": 0.5002,
    }


def _log_mlflow_run(
    args: argparse.Namespace,
    validation,
    validation_artifacts,
    bracket_artifacts,
    elo_kwargs: dict[str, float],
) -> None:
    run_name = args.run_name or f"elo-seed-tuned-{args.league.lower()}"
    tags = {
        "hypothesis": "Tuned carryover Elo may improve the seed-plus-Elo tournament baseline",
        "model_family": "logistic_regression",
        "league": "men" if args.league == "M" else "women",
        "season_window": f"pre-{min(args.holdout_seasons)}",
        "depends_on": "feature:seed_diff_v1,feature:elo_tuned_carryover_v1",
        "retest_if": "tuned Elo parameters or tournament carryover policy change",
        "leakage_audit": "passed",
    }
    with start_tracked_run(run_name, tags=tags):
        mlflow.log_params(
            {
                "league": args.league,
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "features": "seed_diff,elo_diff",
                "elo_day_cutoff": args.day_cutoff,
                **{f"elo_{key}": value for key, value in elo_kwargs.items()},
            }
        )
        mlflow.log_metrics(
            {
                "flat_brier": validation.overall_flat_brier,
                "log_loss": validation.overall_log_loss,
            }
        )
        if "r1_brier" in validation.per_season_metrics.columns:
            r1_values = validation.per_season_metrics["r1_brier"].dropna()
            if not r1_values.empty:
                mlflow.log_metric("r1_brier_mean", float(r1_values.mean()))
        if "r2plus_brier" in validation.per_season_metrics.columns:
            r2_values = validation.per_season_metrics["r2plus_brier"].dropna()
            if not r2_values.empty:
                mlflow.log_metric("r2plus_brier_mean", float(r2_values.mean()))

        for artifact_path in [
            validation_artifacts.per_season_metrics_path,
            validation_artifacts.calibration_table_path,
            validation_artifacts.oof_predictions_path,
            validation_artifacts.reliability_diagram_path,
            bracket_artifacts.play_prob_path,
            bracket_artifacts.slot_team_prob_path,
        ]:
            mlflow.log_artifact(str(artifact_path))
        if bracket_artifacts.bucket_summary_path is not None:
            mlflow.log_artifact(str(bracket_artifacts.bucket_summary_path))
        if bracket_artifacts.round_summary_path is not None:
            mlflow.log_artifact(str(bracket_artifacts.round_summary_path))


if __name__ == "__main__":
    raise SystemExit(main())
