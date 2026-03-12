from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import mlflow
import pandas as pd

from mmlm2026.evaluation.validation import (
    apply_probability_calibrator,
    build_logistic_pipeline,
    collect_training_oof_predictions,
    fit_probability_calibrator,
    save_validation_artifacts,
    validate_season_holdouts_with_calibration,
)
from mmlm2026.features.baseline import (
    build_seed_diff_matchup_features_from_seeds,
    build_seed_diff_tourney_features,
)
from mmlm2026.features.elo import (
    build_elo_seed_matchup_features,
    build_elo_seed_tourney_features,
    compute_end_of_regular_season_elo,
)
from mmlm2026.utils.mlflow_tracking import start_tracked_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VAL-02 calibration audit.")
    parser.add_argument("--model", choices=["seed_diff", "elo_seed"], required=True)
    parser.add_argument("--league", choices=["M", "W"], required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument("--holdout-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/calibration_runs"))
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--day-cutoff", type=int, default=134)
    parser.add_argument("--k-factor", type=float, default=20.0)
    parser.add_argument("--home-advantage", type=float, default=100.0)
    parser.add_argument("--clip-min", type=float, default=0.025)
    parser.add_argument("--clip-max", type=float, default=0.975)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    feature_table, infer_builder, feature_cols = _build_feature_context(args)

    output_dir = args.output_dir / f"{args.model}_{args.league.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_table.to_parquet(output_dir / "feature_table.parquet", index=False)

    audit = validate_season_holdouts_with_calibration(
        feature_table,
        feature_cols=feature_cols,
        holdout_seasons=args.holdout_seasons,
        model_builder=build_logistic_pipeline,
        train_min_games=50,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )

    raw_artifacts = save_validation_artifacts(audit.raw_summary, output_dir=output_dir / "raw")
    isotonic_artifacts = save_validation_artifacts(
        audit.isotonic_summary,
        output_dir=output_dir / "isotonic",
    )
    platt_artifacts = save_validation_artifacts(
        audit.platt_summary,
        output_dir=output_dir / "platt",
    )
    comparison_path = output_dir / "comparison.csv"
    audit.comparison.to_csv(comparison_path, index=False)

    best_method = str(audit.comparison.iloc[0]["method"])
    latest_holdout = max(args.holdout_seasons)
    infer_frame = infer_builder(latest_holdout)
    infer_frame = infer_frame.copy()
    train_frame = feature_table.loc[feature_table["Season"] < latest_holdout].copy()
    base_model = build_logistic_pipeline(feature_cols)
    base_model.fit(train_frame[feature_cols], train_frame["outcome"].astype(int))
    raw_preds = base_model.predict_proba(infer_frame[feature_cols])[:, 1]

    if best_method == "raw":
        infer_frame["Pred"] = raw_preds
    else:
        calibration_oof = collect_training_oof_predictions(
            train_frame,
            feature_cols=feature_cols,
            model_builder=build_logistic_pipeline,
            train_min_games=50,
        )
        calibrator = fit_probability_calibrator(
            calibration_oof["pred"],
            calibration_oof["outcome"],
            method=best_method,
        )
        infer_frame["Pred"] = apply_probability_calibrator(
            raw_preds,
            calibrator,
            method=best_method,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
        )
    infer_frame["ID"] = [
        f"{latest_holdout}_{low}_{high}"
        for low, high in zip(
            infer_frame["LowTeamID"],
            infer_frame["HighTeamID"],
            strict=False,
        )
    ]
    infer_output_path = output_dir / f"{best_method}_inference_preds.csv"
    infer_frame[["ID", "Pred"]].to_csv(infer_output_path, index=False)

    if args.log_mlflow:
        _log_mlflow_run(
            args,
            feature_cols,
            audit,
            comparison_path,
            [
                raw_artifacts.per_season_metrics_path,
                raw_artifacts.calibration_table_path,
                raw_artifacts.oof_predictions_path,
                raw_artifacts.reliability_diagram_path,
                isotonic_artifacts.per_season_metrics_path,
                isotonic_artifacts.calibration_table_path,
                isotonic_artifacts.oof_predictions_path,
                isotonic_artifacts.reliability_diagram_path,
                platt_artifacts.per_season_metrics_path,
                platt_artifacts.calibration_table_path,
                platt_artifacts.oof_predictions_path,
                platt_artifacts.reliability_diagram_path,
                comparison_path,
                infer_output_path,
            ],
            best_method=best_method,
        )

    print(audit.comparison.to_string(index=False))
    print(f"Artifacts written under {output_dir}")
    return 0


def _build_feature_context(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, Callable[[int], pd.DataFrame], list[str]]:
    if args.model == "seed_diff":
        results_path, seeds_path = _seed_diff_paths(args.data_dir, args.league)
        results = pd.read_csv(results_path)
        seeds = pd.read_csv(seeds_path)
        feature_table = build_seed_diff_tourney_features(results, seeds, league=args.league)

        def infer_builder(season: int) -> pd.DataFrame:
            return build_seed_diff_matchup_features_from_seeds(
                seeds,
                season=season,
                league=args.league,
            )

        return feature_table, infer_builder, ["seed_diff"]

    regular_season_path, results_path, seeds_path = _elo_paths(args.data_dir, args.league)
    regular_season = pd.read_csv(regular_season_path)
    results = pd.read_csv(results_path)
    seeds = pd.read_csv(seeds_path)
    elo_ratings = compute_end_of_regular_season_elo(
        regular_season,
        day_cutoff=args.day_cutoff,
        k_factor=args.k_factor,
        home_advantage=args.home_advantage,
    )
    feature_table = build_elo_seed_tourney_features(results, seeds, elo_ratings, league=args.league)

    def infer_builder(season: int) -> pd.DataFrame:
        return build_elo_seed_matchup_features(
            seeds,
            elo_ratings,
            season=season,
            league=args.league,
        )

    return feature_table, infer_builder, ["seed_diff", "elo_diff"]


def _seed_diff_paths(data_dir: Path, league: str) -> tuple[Path, Path]:
    prefix = "M" if league == "M" else "W"
    return (
        data_dir / f"{prefix}NCAATourneyCompactResults.csv",
        data_dir / f"{prefix}NCAATourneySeeds.csv",
    )


def _elo_paths(data_dir: Path, league: str) -> tuple[Path, Path, Path]:
    prefix = "M" if league == "M" else "W"
    return (
        data_dir / f"{prefix}RegularSeasonCompactResults.csv",
        data_dir / f"{prefix}NCAATourneyCompactResults.csv",
        data_dir / f"{prefix}NCAATourneySeeds.csv",
    )


def _log_mlflow_run(
    args: argparse.Namespace,
    feature_cols: list[str],
    audit,
    comparison_path: Path,
    artifact_paths: list[Path],
    *,
    best_method: str,
) -> None:
    run_name = args.run_name or f"val-02-{args.model}-{args.league.lower()}"
    tags = {
        "hypothesis": "Training-fold calibration may improve held-out tournament Brier",
        "model_family": "calibration_audit",
        "league": "men" if args.league == "M" else "women",
        "season_window": f"pre-{min(args.holdout_seasons)}",
        "depends_on": f"model:{args.model}",
        "retest_if": "base model family or calibration policy changes",
        "leakage_audit": "passed",
    }
    with start_tracked_run(run_name, tags=tags):
        mlflow.log_params(
            {
                "league": args.league,
                "base_model": args.model,
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "features": ",".join(feature_cols),
                "best_calibration_method": best_method,
                "clip_min": args.clip_min,
                "clip_max": args.clip_max,
            }
        )
        for _, row in audit.comparison.iterrows():
            method = str(row["method"])
            mlflow.log_metric(f"{method}_flat_brier", float(row["flat_brier"]))
            mlflow.log_metric(f"{method}_log_loss", float(row["log_loss"]))
            mlflow.log_metric(f"{method}_ece", float(row["ece"]))
            mlflow.log_metric(f"{method}_max_abs_gap", float(row["max_abs_gap"]))
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))


if __name__ == "__main__":
    raise SystemExit(main())
