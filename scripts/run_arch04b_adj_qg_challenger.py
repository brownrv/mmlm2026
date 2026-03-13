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
from mmlm2026.features.elo import compute_pre_tourney_elo_ratings
from mmlm2026.features.primary import (
    build_phase_ab_matchup_features,
    build_phase_ab_team_features,
    build_phase_ab_tourney_features,
)
from mmlm2026.submission.frozen_models import WOMEN_ELO_PARAMS
from mmlm2026.utils.mlflow_tracking import start_tracked_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ARCH-04B plus women_hca_adj_qg_diff challenger."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument("--holdout-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/challenger_runs"))
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--run-name", default="arch-04c-arch04b-plus-adj-qg-women")
    parser.add_argument("--day-cutoff", type=int, default=134)
    parser.add_argument("--season-floor", type=int, default=2010)
    parser.add_argument("--women-hca", type=float, default=3.0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    regular_season = pd.read_csv(args.data_dir / "WRegularSeasonCompactResults.csv")
    regular_season_detailed = pd.read_csv(args.data_dir / "WRegularSeasonDetailedResults.csv")
    results = pd.read_csv(args.data_dir / "WNCAATourneyCompactResults.csv")
    seeds = pd.read_csv(args.data_dir / "WNCAATourneySeeds.csv")
    slots = pd.read_csv(args.data_dir / "WNCAATourneySlots.csv")

    regular_season = regular_season.loc[regular_season["Season"] >= args.season_floor].copy()
    regular_season_detailed = regular_season_detailed.loc[
        regular_season_detailed["Season"] >= args.season_floor
    ].copy()
    results = results.loc[results["Season"] >= args.season_floor].copy()
    seeds = seeds.loc[seeds["Season"] >= args.season_floor].copy()
    slots = slots.loc[slots["Season"] >= args.season_floor].copy()

    elo_ratings = compute_pre_tourney_elo_ratings(
        regular_season,
        tourney_results=results,
        day_cutoff=args.day_cutoff,
        **WOMEN_ELO_PARAMS,
    )
    team_features = build_phase_ab_team_features(
        regular_season,
        regular_season_detailed,
        elo_ratings,
        day_cutoff=args.day_cutoff,
        women_hca_adjustment=args.women_hca,
    )
    feature_table = build_phase_ab_tourney_features(results, seeds, team_features, league="W")
    feature_cols = ["seed_diff", "elo_diff", "women_hca_adj_qg_diff"]

    output_dir = args.output_dir / "w_arch04b_plus_adj_qg"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_table.to_parquet(output_dir / "women_arch04b_adj_qg_features.parquet", index=False)
        team_features.to_parquet(output_dir / "phase_ab_team_features.parquet", index=False)
        elo_ratings.to_parquet(output_dir / "elo_ratings.parquet", index=False)

    validation = validate_season_holdouts(
        feature_table,
        feature_cols=feature_cols,
        holdout_seasons=args.holdout_seasons,
        train_min_games=50,
    )
    validation_artifacts = save_validation_artifacts(
        validation,
        output_dir=output_dir / "validation",
    )

    latest_holdout = max(args.holdout_seasons)
    train_frame = feature_table.loc[feature_table["Season"] < latest_holdout].copy()
    infer_frame = build_phase_ab_matchup_features(
        seeds,
        team_features,
        season=latest_holdout,
        league="W",
    )
    model = build_logistic_pipeline(feature_cols)
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
        _log_mlflow_run(args, validation, validation_artifacts, bracket_artifacts)

    print(validation.per_season_metrics.to_string(index=False))
    print(
        "\nOverall metrics:",
        f"flat_brier={validation.overall_flat_brier:.6f}",
        f"log_loss={validation.overall_log_loss:.6f}",
    )
    print(f"Artifacts written under {output_dir}")
    return 0


def _log_mlflow_run(
    args: argparse.Namespace,
    validation,
    validation_artifacts,
    bracket_artifacts,
) -> None:
    tags = {
        "hypothesis": (
            "The tuned ARCH-04B women baseline may improve if it absorbs the "
            "women_hca_adj_qg_diff strength signal without changing model family"
        ),
        "model_family": "logistic_regression",
        "league": "women",
        "season_window": f"pre-{min(args.holdout_seasons)}",
        "depends_on": "arch-04b,feature:feat_13_women_hca_adj_qg_v1",
        "retest_if": "women HCA-adjusted quality-gap definition changes",
        "leakage_audit": "passed",
    }
    with start_tracked_run(args.run_name, tags=tags):
        mlflow.log_params(
            {
                "league": "W",
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "features": "seed_diff,elo_diff,women_hca_adj_qg_diff",
                "season_floor": args.season_floor,
                "elo_day_cutoff": args.day_cutoff,
                "women_hca": args.women_hca,
                **{f"elo_{key}": value for key, value in WOMEN_ELO_PARAMS.items()},
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
