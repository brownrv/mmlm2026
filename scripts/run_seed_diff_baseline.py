from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts
from mmlm2026.evaluation.validation import save_validation_artifacts, validate_season_holdouts
from mmlm2026.features.baseline import build_seed_diff_tourney_features
from mmlm2026.utils.mlflow_tracking import start_tracked_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the seed-diff tournament baseline.")
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
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    results_path, seeds_path, slots_path = _league_paths(args.data_dir, args.league)
    results = pd.read_csv(results_path)
    seeds = pd.read_csv(seeds_path)
    slots = pd.read_csv(slots_path)

    feature_table = build_seed_diff_tourney_features(results, seeds, league=args.league)

    output_dir = args.output_dir / args.league.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_table.to_parquet(output_dir / "seed_diff_features.parquet", index=False)

    validation = validate_season_holdouts(
        feature_table,
        feature_cols=["seed_diff"],
        holdout_seasons=args.holdout_seasons,
        train_min_games=50,
    )
    validation_artifacts = save_validation_artifacts(
        validation,
        output_dir=output_dir / "validation",
    )

    latest_holdout = max(args.holdout_seasons)
    holdout_predictions = validation.oof_predictions.loc[
        validation.oof_predictions["Season"] == latest_holdout
    ].copy()
    holdout_predictions["ID"] = [
        f"{latest_holdout}_{low}_{high}"
        for low, high in zip(
            holdout_predictions["LowTeamID"],
            holdout_predictions["HighTeamID"],
            strict=False,
        )
    ]
    holdout_predictions["Pred"] = holdout_predictions["pred"]
    bracket = compute_bracket_diagnostics(
        slots,
        seeds,
        holdout_predictions[["ID", "Pred"]],
        season=latest_holdout,
        results=results,
        round_col="Round",
    )
    bracket_artifacts = save_bracket_artifacts(bracket, output_dir=output_dir / "bracket")

    if args.log_mlflow:
        _log_mlflow_run(
            args,
            validation,
            validation_artifacts,
            bracket_artifacts,
        )

    print(validation.per_season_metrics.to_string(index=False))
    print(
        "\nOverall metrics:",
        f"flat_brier={validation.overall_flat_brier:.6f}",
        f"log_loss={validation.overall_log_loss:.6f}",
    )
    print(f"Artifacts written under {output_dir}")
    return 0


def _league_paths(data_dir: Path, league: str) -> tuple[Path, Path, Path]:
    prefix = "M" if league == "M" else "W"
    results_path = data_dir / f"{prefix}NCAATourneyCompactResults.csv"
    seeds_path = data_dir / f"{prefix}NCAATourneySeeds.csv"
    slots_path = data_dir / f"{prefix}NCAATourneySlots.csv"
    return results_path, seeds_path, slots_path


def _log_mlflow_run(
    args: argparse.Namespace,
    validation,
    validation_artifacts,
    bracket_artifacts,
) -> None:
    run_name = args.run_name or f"seed-diff-baseline-{args.league.lower()}"
    tags = {
        "hypothesis": "seed difference alone is a strong tournament baseline",
        "model_family": "logistic_regression",
        "league": "men" if args.league == "M" else "women",
        "season_window": f"pre-{min(args.holdout_seasons)}",
        "depends_on": "feature:seed_diff_v1",
        "retest_if": "tournament seed assignment rules change",
        "leakage_audit": "passed",
    }
    with start_tracked_run(run_name, tags=tags):
        mlflow.log_params(
            {
                "league": args.league,
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "feature": "seed_diff",
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
