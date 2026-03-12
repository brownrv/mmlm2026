from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts
from mmlm2026.evaluation.ensemble import (
    blend_predictions,
    find_best_blend_weight,
    merge_prediction_frames,
)
from mmlm2026.evaluation.validation import (
    ValidationSummary,
    build_calibration_table,
    build_hist_gbt_pipeline,
    build_logistic_pipeline,
    collect_training_oof_predictions,
    save_validation_artifacts,
)
from mmlm2026.features.elo import (
    build_elo_seed_matchup_features,
    build_elo_seed_tourney_features,
    compute_end_of_regular_season_elo,
)
from mmlm2026.features.primary import (
    add_phase_c_features,
    build_phase_ab_matchup_features,
    build_phase_ab_team_features,
    build_phase_ab_tourney_features,
    phase_abc_feature_columns,
)
from mmlm2026.utils.mlflow_tracking import start_tracked_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simple blend experiments.")
    parser.add_argument("--league", choices=["M", "W"], required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument("--holdout-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/combo_runs"))
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--day-cutoff", type=int, default=134)
    parser.add_argument("--ranking-day-cutoff", type=int, default=133)
    parser.add_argument("--k-factor", type=float, default=20.0)
    parser.add_argument("--home-advantage", type=float, default=100.0)
    parser.add_argument("--season-floor", type=int, default=2010)
    parser.add_argument("--weight-step", type=float, default=0.05)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    context = _build_context(args)
    output_dir = args.output_dir / args.league.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        context["logistic_frame"].to_parquet(output_dir / "logistic_frame.parquet", index=False)
        context["tree_frame"].to_parquet(output_dir / "tree_frame.parquet", index=False)

    metrics_rows: list[dict[str, float | int | None]] = []
    prediction_frames: list[pd.DataFrame] = []
    weight_rows: list[dict[str, float | int]] = []

    for holdout_season in args.holdout_seasons:
        logistic_train = (
            context["logistic_frame"]
            .loc[context["logistic_frame"]["Season"] < holdout_season]
            .copy()
        )
        logistic_valid = (
            context["logistic_frame"]
            .loc[context["logistic_frame"]["Season"] == holdout_season]
            .copy()
        )
        tree_train = (
            context["tree_frame"].loc[context["tree_frame"]["Season"] < holdout_season].copy()
        )
        tree_valid = (
            context["tree_frame"].loc[context["tree_frame"]["Season"] == holdout_season].copy()
        )

        logistic_oof = collect_training_oof_predictions(
            logistic_train,
            feature_cols=context["logistic_feature_cols"],
            model_builder=build_logistic_pipeline,
            train_min_games=50,
        )
        tree_oof = collect_training_oof_predictions(
            tree_train,
            feature_cols=context["tree_feature_cols"],
            model_builder=build_hist_gbt_pipeline,
            train_min_games=50,
        )
        merged_oof = merge_prediction_frames(logistic_oof, tree_oof)
        best_weight, grid = find_best_blend_weight(
            merged_oof["pred_first"],
            merged_oof["pred_second"],
            merged_oof["outcome"],
            step=args.weight_step,
        )
        weight_rows.extend(
            {
                "holdout_season": holdout_season,
                "first_weight": float(row["first_weight"]),
                "flat_brier": float(row["flat_brier"]),
            }
            for row in grid.to_dict(orient="records")
        )

        logistic_model = build_logistic_pipeline(context["logistic_feature_cols"])
        logistic_model.fit(
            logistic_train[context["logistic_feature_cols"]],
            logistic_train["outcome"].astype(int),
        )
        tree_model = build_hist_gbt_pipeline(context["tree_feature_cols"])
        tree_model.fit(
            tree_train[context["tree_feature_cols"]],
            tree_train["outcome"].astype(int),
        )

        logistic_preds = logistic_model.predict_proba(
            logistic_valid[context["logistic_feature_cols"]]
        )[:, 1]
        tree_preds = tree_model.predict_proba(tree_valid[context["tree_feature_cols"]])[:, 1]
        blend_preds = blend_predictions(logistic_preds, tree_preds, first_weight=best_weight)

        valid_frame = logistic_valid[
            ["Season", "LowTeamID", "HighTeamID", "outcome", "round_group"]
        ].copy()
        valid_frame["pred"] = blend_preds.to_numpy()
        prediction_frames.append(valid_frame)

        metrics_rows.append(
            _build_metrics_row(
                holdout_season=holdout_season,
                train_games=len(logistic_train),
                valid_frame=valid_frame,
                best_weight=best_weight,
            )
        )

    summary = _build_summary(metrics_rows, prediction_frames)
    validation_artifacts = save_validation_artifacts(summary, output_dir=output_dir / "validation")
    weights_path = output_dir / "blend_weight_grid.csv"
    pd.DataFrame(weight_rows).to_csv(weights_path, index=False)

    latest_holdout = max(args.holdout_seasons)
    train_logistic = (
        context["logistic_frame"].loc[context["logistic_frame"]["Season"] < latest_holdout].copy()
    )
    train_tree = context["tree_frame"].loc[context["tree_frame"]["Season"] < latest_holdout].copy()
    latest_weight = (
        pd.DataFrame(weight_rows)
        .loc[lambda df: df["holdout_season"] == latest_holdout]
        .sort_values(["flat_brier", "first_weight"])
        .iloc[0]["first_weight"]
    )

    logistic_model = build_logistic_pipeline(context["logistic_feature_cols"])
    logistic_model.fit(
        train_logistic[context["logistic_feature_cols"]],
        train_logistic["outcome"].astype(int),
    )
    tree_model = build_hist_gbt_pipeline(context["tree_feature_cols"])
    tree_model.fit(
        train_tree[context["tree_feature_cols"]],
        train_tree["outcome"].astype(int),
    )

    infer_logistic = context["logistic_infer_builder"](latest_holdout)
    infer_tree = context["tree_infer_builder"](latest_holdout)
    logistic_infer_preds = logistic_model.predict_proba(
        infer_logistic[context["logistic_feature_cols"]]
    )[:, 1]
    tree_infer_preds = tree_model.predict_proba(infer_tree[context["tree_feature_cols"]])[:, 1]
    infer_logistic["Pred"] = blend_predictions(
        logistic_infer_preds,
        tree_infer_preds,
        first_weight=float(latest_weight),
    ).to_numpy()
    infer_logistic["ID"] = [
        f"{latest_holdout}_{low}_{high}"
        for low, high in zip(
            infer_logistic["LowTeamID"],
            infer_logistic["HighTeamID"],
            strict=False,
        )
    ]

    bracket = compute_bracket_diagnostics(
        context["slots"],
        context["seeds"],
        infer_logistic[["ID", "Pred"]],
        season=latest_holdout,
        results=context["results"],
        round_col="Round",
    )
    bracket_artifacts = save_bracket_artifacts(bracket, output_dir=output_dir / "bracket")

    if args.log_mlflow:
        _log_mlflow_run(
            args,
            summary,
            validation_artifacts,
            bracket_artifacts,
            weights_path,
            latest_weight=float(latest_weight),
        )

    print(summary.per_season_metrics.to_string(index=False))
    print(
        "\nOverall metrics:",
        f"flat_brier={summary.overall_flat_brier:.6f}",
        f"log_loss={summary.overall_log_loss:.6f}",
    )
    print(f"Artifacts written under {output_dir}")
    return 0


def _build_context(args: argparse.Namespace) -> dict[str, object]:
    prefix = "M" if args.league == "M" else "W"
    data_dir = args.data_dir
    regular_season = pd.read_csv(data_dir / f"{prefix}RegularSeasonCompactResults.csv")
    regular_season_detailed = pd.read_csv(data_dir / f"{prefix}RegularSeasonDetailedResults.csv")
    results = pd.read_csv(data_dir / f"{prefix}NCAATourneyCompactResults.csv")
    seeds = pd.read_csv(data_dir / f"{prefix}NCAATourneySeeds.csv")
    slots = pd.read_csv(data_dir / f"{prefix}NCAATourneySlots.csv")
    massey = pd.read_csv(data_dir / "MMasseyOrdinals.csv") if args.league == "M" else None

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
    logistic_frame = build_elo_seed_tourney_features(
        results,
        seeds,
        elo_ratings,
        league=args.league,
    )

    team_features = build_phase_ab_team_features(
        regular_season,
        regular_season_detailed,
        elo_ratings,
        massey_ordinals=massey,
        day_cutoff=args.day_cutoff,
        ranking_day_cutoff=args.ranking_day_cutoff,
    )
    tree_frame = add_phase_c_features(
        build_phase_ab_tourney_features(results, seeds, team_features, league=args.league)
    )

    def logistic_infer_builder(season: int) -> pd.DataFrame:
        return build_elo_seed_matchup_features(
            seeds,
            elo_ratings,
            season=season,
            league=args.league,
        )

    def tree_infer_builder(season: int) -> pd.DataFrame:
        return add_phase_c_features(
            build_phase_ab_matchup_features(
                seeds,
                team_features,
                season=season,
                league=args.league,
            )
        )

    return {
        "results": results,
        "seeds": seeds,
        "slots": slots,
        "logistic_frame": logistic_frame,
        "tree_frame": tree_frame,
        "logistic_feature_cols": ["seed_diff", "elo_diff"],
        "tree_feature_cols": phase_abc_feature_columns(args.league),
        "logistic_infer_builder": logistic_infer_builder,
        "tree_infer_builder": tree_infer_builder,
    }


def _build_metrics_row(
    *,
    holdout_season: int,
    train_games: int,
    valid_frame: pd.DataFrame,
    best_weight: float,
) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {
        "Season": holdout_season,
        "train_tourney_games": train_games,
        "valid_tourney_games": len(valid_frame),
        "flat_brier": float(brier_score_loss(valid_frame["outcome"], valid_frame["pred"])),
        "log_loss": float(log_loss(valid_frame["outcome"], valid_frame["pred"], labels=[0, 1])),
        "first_weight": best_weight,
    }
    r1 = valid_frame.loc[valid_frame["round_group"] == "R1"]
    if not r1.empty:
        row["r1_brier"] = float(brier_score_loss(r1["outcome"], r1["pred"]))
    r2 = valid_frame.loc[valid_frame["round_group"] == "R2+"]
    if not r2.empty:
        row["r2plus_brier"] = float(brier_score_loss(r2["outcome"], r2["pred"]))
    return row


def _build_summary(
    metrics_rows: list[dict[str, float | int | None]],
    prediction_frames: list[pd.DataFrame],
) -> ValidationSummary:
    per_season_metrics = pd.DataFrame(metrics_rows).sort_values("Season").reset_index(drop=True)
    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    calibration_table = build_calibration_table(
        oof_predictions["outcome"].astype(int),
        oof_predictions["pred"],
        n_bins=10,
    )
    return ValidationSummary(
        per_season_metrics=per_season_metrics,
        calibration_table=calibration_table,
        oof_predictions=oof_predictions,
    )


def _log_mlflow_run(
    args: argparse.Namespace,
    summary: ValidationSummary,
    validation_artifacts,
    bracket_artifacts,
    weights_path: Path,
    *,
    latest_weight: float,
) -> None:
    run_name = args.run_name or f"combo-simple-blend-{args.league.lower()}"
    combo_id = "COMBO-01" if args.league == "M" else "COMBO-02"
    tags = {
        "hypothesis": "A simple blend of Elo logistic and GBT may improve tournament Brier",
        "model_family": "simple_blend",
        "league": "men" if args.league == "M" else "women",
        "season_window": f"pre-{min(args.holdout_seasons)}",
        "depends_on": "arch-03,arch-07" if args.league == "M" else "arch-04,arch-08",
        "retest_if": "blend members or weight-search policy change",
        "leakage_audit": "passed",
    }
    with start_tracked_run(run_name, tags=tags):
        mlflow.log_params(
            {
                "combo_id": combo_id,
                "league": args.league,
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "weight_step": args.weight_step,
                "latest_holdout_first_weight": latest_weight,
            }
        )
        mlflow.log_metrics(
            {
                "flat_brier": summary.overall_flat_brier,
                "log_loss": summary.overall_log_loss,
            }
        )
        if "r1_brier" in summary.per_season_metrics.columns:
            r1_values = summary.per_season_metrics["r1_brier"].dropna()
            if not r1_values.empty:
                mlflow.log_metric("r1_brier_mean", float(r1_values.mean()))
        if "r2plus_brier" in summary.per_season_metrics.columns:
            r2_values = summary.per_season_metrics["r2plus_brier"].dropna()
            if not r2_values.empty:
                mlflow.log_metric("r2plus_brier_mean", float(r2_values.mean()))
        for artifact_path in [
            validation_artifacts.per_season_metrics_path,
            validation_artifacts.calibration_table_path,
            validation_artifacts.oof_predictions_path,
            validation_artifacts.reliability_diagram_path,
            weights_path,
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
