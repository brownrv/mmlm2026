from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts
from mmlm2026.evaluation.validation import (
    build_xgboost_pipeline,
    save_validation_artifacts,
    validate_season_holdouts_with_model,
)
from mmlm2026.features.elo import compute_end_of_regular_season_elo
from mmlm2026.features.primary import (
    add_phase_c_features,
    build_phase_ab_matchup_features,
    build_phase_ab_team_features,
    build_phase_ab_tourney_features,
    phase_abc_feature_columns,
)
from mmlm2026.utils.mlflow_tracking import start_tracked_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase A+B+C XGBoost primary model.")
    parser.add_argument("--league", choices=["M", "W"], required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument("--holdout-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/primary_runs"))
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--day-cutoff", type=int, default=134)
    parser.add_argument("--ranking-day-cutoff", type=int, default=133)
    parser.add_argument("--k-factor", type=float, default=20.0)
    parser.add_argument("--home-advantage", type=float, default=100.0)
    parser.add_argument("--season-floor", type=int, default=2010)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    (
        regular_season_path,
        regular_season_detailed_path,
        tourney_results_path,
        seeds_path,
        slots_path,
        massey_path,
    ) = _league_paths(args.data_dir, args.league)

    regular_season = pd.read_csv(regular_season_path)
    regular_season_detailed = pd.read_csv(regular_season_detailed_path)
    results = pd.read_csv(tourney_results_path)
    seeds = pd.read_csv(seeds_path)
    slots = pd.read_csv(slots_path)
    massey = pd.read_csv(massey_path) if massey_path is not None else None

    regular_season = regular_season.loc[regular_season["Season"] >= args.season_floor].copy()
    regular_season_detailed = regular_season_detailed.loc[
        regular_season_detailed["Season"] >= args.season_floor
    ].copy()
    results = results.loc[results["Season"] >= args.season_floor].copy()
    seeds = seeds.loc[seeds["Season"] >= args.season_floor].copy()
    slots = slots.loc[slots["Season"] >= args.season_floor].copy()
    if massey is not None:
        massey = massey.loc[massey["Season"] >= args.season_floor].copy()

    elo_ratings = compute_end_of_regular_season_elo(
        regular_season,
        day_cutoff=args.day_cutoff,
        k_factor=args.k_factor,
        home_advantage=args.home_advantage,
    )
    team_features = build_phase_ab_team_features(
        regular_season,
        regular_season_detailed,
        elo_ratings,
        massey_ordinals=massey,
        day_cutoff=args.day_cutoff,
        ranking_day_cutoff=args.ranking_day_cutoff,
    )
    feature_table = add_phase_c_features(
        build_phase_ab_tourney_features(
            results,
            seeds,
            team_features,
            league=args.league,
        )
    )
    feature_cols = phase_abc_feature_columns(args.league)

    output_dir = args.output_dir / f"{args.league.lower()}_phase_abc_xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_table.to_parquet(output_dir / "phase_abc_tourney_features.parquet", index=False)
        team_features.to_parquet(output_dir / "phase_ab_team_features.parquet", index=False)
        elo_ratings.to_parquet(output_dir / "elo_ratings.parquet", index=False)

    validation = validate_season_holdouts_with_model(
        feature_table,
        feature_cols=feature_cols,
        holdout_seasons=args.holdout_seasons,
        train_min_games=50,
        model_builder=build_xgboost_pipeline,
    )
    validation_artifacts = save_validation_artifacts(
        validation,
        output_dir=output_dir / "validation",
    )

    latest_holdout = max(args.holdout_seasons)
    train_frame = feature_table.loc[feature_table["Season"] < latest_holdout].copy()
    infer_frame = add_phase_c_features(
        build_phase_ab_matchup_features(
            seeds,
            team_features,
            season=latest_holdout,
            league=args.league,
        )
    )
    model = build_xgboost_pipeline(feature_cols)
    model.fit(train_frame[feature_cols], train_frame["outcome"].astype(int))
    infer_frame["Pred"] = model.predict_proba(infer_frame[feature_cols])[:, 1]
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
        _log_mlflow_run(
            args,
            feature_cols,
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


def _league_paths(data_dir: Path, league: str) -> tuple[Path, Path, Path, Path, Path, Path | None]:
    prefix = "M" if league == "M" else "W"
    regular_season_path = data_dir / f"{prefix}RegularSeasonCompactResults.csv"
    regular_season_detailed_path = data_dir / f"{prefix}RegularSeasonDetailedResults.csv"
    tourney_results_path = data_dir / f"{prefix}NCAATourneyCompactResults.csv"
    seeds_path = data_dir / f"{prefix}NCAATourneySeeds.csv"
    slots_path = data_dir / f"{prefix}NCAATourneySlots.csv"
    massey_path = data_dir / "MMasseyOrdinals.csv" if league == "M" else None
    return (
        regular_season_path,
        regular_season_detailed_path,
        tourney_results_path,
        seeds_path,
        slots_path,
        massey_path,
    )


def _log_mlflow_run(
    args: argparse.Namespace,
    feature_cols: list[str],
    validation,
    validation_artifacts,
    bracket_artifacts,
) -> None:
    run_name = args.run_name or f"phase-abc-xgboost-{args.league.lower()}"
    tags = {
        "hypothesis": (
            "XGBoost on Phase A+B+C features improves tournament Brier "
            "over the earlier tree and linear baselines"
        ),
        "model_family": "xgboost",
        "league": "men" if args.league == "M" else "women",
        "season_window": f"pre-{min(args.holdout_seasons)}",
        "depends_on": "feature:seed_diff_v1,feature:elo_v1,feature:phase_b_v1,feature:phase_c_v1",
        "retest_if": "phase A/B/C definitions or xgboost hyperparameters change",
        "leakage_audit": "passed",
    }
    with start_tracked_run(run_name, tags=tags):
        mlflow.log_params(
            {
                "league": args.league,
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "features": ",".join(feature_cols),
                "elo_day_cutoff": args.day_cutoff,
                "massey_day_cutoff": args.ranking_day_cutoff,
                "elo_k_factor": args.k_factor,
                "elo_home_advantage": args.home_advantage,
                "season_floor": args.season_floor,
                "tree_model": "XGBClassifier",
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
