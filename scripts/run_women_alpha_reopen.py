from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import log_loss  # type: ignore[import-untyped]

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts
from mmlm2026.evaluation.validation import (
    ValidationSummary,
    build_calibration_table,
    build_logistic_pipeline,
    save_validation_artifacts,
)
from mmlm2026.evaluation.women_alpha import select_alpha_from_prior_seasons
from mmlm2026.features.elo import (
    compute_pre_tourney_elo_ratings,
    elo_probability_from_diff,
)
from mmlm2026.features.primary import (
    build_phase_ab_matchup_features,
    build_phase_ab_team_features,
    build_phase_ab_tourney_features,
)
from mmlm2026.utils.mlflow_tracking import start_tracked_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the women reference-style model with reopened Elo alpha."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument("--holdout-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/feature_runs"))
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--run-name", default="val-05-women-alpha-reopen")
    parser.add_argument("--day-cutoff", type=int, default=134)
    parser.add_argument("--season-floor", type=int, default=2010)
    parser.add_argument("--women-hca", type=float, default=3.0)
    parser.add_argument("--alpha-fallback", type=float, default=0.10)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument("--alpha-calibration-seasons", type=int, default=8)
    parser.add_argument("--elo-initial-rating", type=float, default=1213.0)
    parser.add_argument("--elo-k-factor", type=float, default=136.0)
    parser.add_argument("--elo-home-advantage", type=float, default=119.0)
    parser.add_argument("--elo-season-carryover", type=float, default=0.9840)
    parser.add_argument("--elo-scale", type=float, default=1888.27)
    parser.add_argument("--elo-mov-alpha", type=float, default=11.4612)
    parser.add_argument("--elo-weight-regular", type=float, default=1.4974)
    parser.add_argument("--elo-weight-tourney", type=float, default=0.5002)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    data_dir = args.data_dir
    regular_season = pd.read_csv(data_dir / "WRegularSeasonCompactResults.csv")
    regular_season_detailed = pd.read_csv(data_dir / "WRegularSeasonDetailedResults.csv")
    results = pd.read_csv(data_dir / "WNCAATourneyCompactResults.csv")
    seeds = pd.read_csv(data_dir / "WNCAATourneySeeds.csv")
    slots = pd.read_csv(data_dir / "WNCAATourneySlots.csv")

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
        initial_rating=args.elo_initial_rating,
        k_factor=args.elo_k_factor,
        home_advantage=args.elo_home_advantage,
        season_carryover=args.elo_season_carryover,
        scale=args.elo_scale,
        mov_alpha=args.elo_mov_alpha,
        weight_regular=args.elo_weight_regular,
        weight_tourney=args.elo_weight_tourney,
    )
    team_features = build_phase_ab_team_features(
        regular_season,
        regular_season_detailed,
        elo_ratings,
        day_cutoff=args.day_cutoff,
        women_hca_adjustment=args.women_hca,
    )
    feature_table = build_phase_ab_tourney_features(
        results,
        seeds,
        team_features,
        league="W",
    )
    feature_cols = ["women_hca_adj_qg_diff", "seed_diff", "close_win_pct_diff"]

    output_dir = args.output_dir / "w_alpha_reopen"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_table.to_parquet(output_dir / "women_alpha_reopen_features.parquet", index=False)
        team_features.to_parquet(output_dir / "phase_ab_team_features.parquet", index=False)

    summary = _validate_holdouts(
        feature_table,
        feature_cols=feature_cols,
        holdout_seasons=args.holdout_seasons,
        elo_scale=args.elo_scale,
        alpha_fallback=args.alpha_fallback,
        alpha_step=args.alpha_step,
        alpha_calibration_seasons=args.alpha_calibration_seasons,
    )
    validation_artifacts = save_validation_artifacts(
        summary,
        output_dir=output_dir / "validation",
    )

    latest_holdout = max(args.holdout_seasons)
    alpha_for_inference = _fit_alpha_from_prior_seasons(
        frame=feature_table,
        feature_cols=feature_cols,
        target_season=latest_holdout,
        elo_scale=args.elo_scale,
        alpha_fallback=args.alpha_fallback,
        alpha_step=args.alpha_step,
        alpha_calibration_seasons=args.alpha_calibration_seasons,
    )
    train_frame = feature_table.loc[feature_table["Season"] < latest_holdout].copy()
    infer_frame = build_phase_ab_matchup_features(
        seeds,
        team_features,
        season=latest_holdout,
        league="W",
    )
    model = build_logistic_pipeline(feature_cols)
    model.fit(train_frame[feature_cols], train_frame["outcome"].astype(int))
    base_pred = pd.Series(
        model.predict_proba(infer_frame[feature_cols])[:, 1],
        index=infer_frame.index,
        dtype=float,
    )
    elo_pred = elo_probability_from_diff(infer_frame["elo_diff"], scale=args.elo_scale)
    infer_frame["Pred"] = (
        (1.0 - alpha_for_inference) * base_pred + alpha_for_inference * elo_pred
    ).clip(
        lower=0.025,
        upper=0.975,
    )
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
            summary,
            validation_artifacts,
            bracket_artifacts,
            alpha_for_inference,
        )

    print(summary.per_season_metrics.to_string(index=False))
    print(f"\nSelected inference alpha={alpha_for_inference:.4f}")
    print(
        "\nOverall metrics:",
        f"flat_brier={summary.overall_flat_brier:.6f}",
        f"log_loss={summary.overall_log_loss:.6f}",
    )
    print(f"Artifacts written under {output_dir}")
    return 0


def _validate_holdouts(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    holdout_seasons: list[int],
    elo_scale: float,
    alpha_fallback: float,
    alpha_step: float,
    alpha_calibration_seasons: int,
    train_min_games: int = 50,
) -> ValidationSummary:
    work = frame.loc[frame["outcome"].notna()].copy()
    metrics_rows: list[dict[str, float | int | None]] = []
    prediction_frames: list[pd.DataFrame] = []

    for holdout_season in holdout_seasons:
        train = work.loc[work["Season"] < holdout_season].copy()
        valid = work.loc[work["Season"] == holdout_season].copy()
        if len(train) < train_min_games:
            raise ValueError(
                f"Training set for season {holdout_season} has {len(train)} games; "
                f"minimum required is {train_min_games}."
            )
        alpha = _fit_alpha_from_prior_seasons(
            frame=work,
            feature_cols=feature_cols,
            target_season=holdout_season,
            elo_scale=elo_scale,
            alpha_fallback=alpha_fallback,
            alpha_step=alpha_step,
            alpha_calibration_seasons=alpha_calibration_seasons,
        )
        model = build_logistic_pipeline(feature_cols)
        model.fit(train[feature_cols], train["outcome"].astype(int))

        base_pred = pd.Series(
            model.predict_proba(valid[feature_cols])[:, 1],
            index=valid.index,
            dtype=float,
        )
        elo_pred = elo_probability_from_diff(valid["elo_diff"], scale=elo_scale)
        pred = ((1.0 - alpha) * base_pred + alpha * elo_pred).clip(
            lower=0.025,
            upper=0.975,
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "Season": valid["Season"].to_numpy(),
                    "LowTeamID": valid["LowTeamID"].to_numpy(),
                    "HighTeamID": valid["HighTeamID"].to_numpy(),
                    "outcome": valid["outcome"].astype(int).to_numpy(),
                    "pred": pred.to_numpy(),
                    "round_group": valid["round_group"].to_numpy(),
                }
            )
        )

        season_pred = prediction_frames[-1]
        metrics_rows.append(
            {
                "Season": holdout_season,
                "train_tourney_games": len(train),
                "valid_tourney_games": len(valid),
                "selected_alpha": alpha,
                "flat_brier": float(((season_pred["pred"] - season_pred["outcome"]) ** 2).mean()),
                "log_loss": float(log_loss(season_pred["outcome"], season_pred["pred"])),
                "r1_brier": _round_brier(season_pred, "R1"),
                "r2plus_brier": _round_brier(season_pred, "R2+"),
            }
        )

    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    calibration_table = build_calibration_table(
        oof_predictions["outcome"],
        oof_predictions["pred"],
        n_bins=10,
    )
    return ValidationSummary(
        per_season_metrics=pd.DataFrame(metrics_rows).sort_values("Season").reset_index(drop=True),
        calibration_table=calibration_table,
        oof_predictions=oof_predictions,
    )


def _fit_alpha_from_prior_seasons(
    *,
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_season: int,
    elo_scale: float,
    alpha_fallback: float,
    alpha_step: float,
    alpha_calibration_seasons: int,
    train_min_games: int = 50,
) -> float:
    return select_alpha_from_prior_seasons(
        frame=frame,
        feature_cols=feature_cols,
        target_season=target_season,
        elo_scale=elo_scale,
        alpha_fallback=alpha_fallback,
        alpha_step=alpha_step,
        alpha_calibration_seasons=alpha_calibration_seasons,
        train_min_games=train_min_games,
    )


def _round_brier(frame: pd.DataFrame, round_group: str) -> float | None:
    mask = frame["round_group"] == round_group
    if not mask.any():
        return None
    return float(((frame.loc[mask, "pred"] - frame.loc[mask, "outcome"]) ** 2).mean())


def _log_mlflow_run(
    args: argparse.Namespace,
    summary: ValidationSummary,
    validation_artifacts,
    bracket_artifacts,
    inference_alpha: float,
) -> None:
    tags = {
        "hypothesis": (
            "Reopening the women Elo blend weight may recover complementary signal "
            "that the fixed 10 percent blend is suppressing"
        ),
        "model_family": "logistic_regression_plus_reestimated_elo_blend",
        "league": "women",
        "season_window": f"pre-{min(args.holdout_seasons)}",
        "depends_on": (
            "arch-04b,feature:feat_13_women_hca_adj_qg_v1,feature:feat_15_women_closewr3_v1"
        ),
        "retest_if": "women alpha calibration or Elo component changes",
        "leakage_audit": "passed",
    }
    with start_tracked_run(args.run_name, tags=tags):
        mlflow.log_params(
            {
                "league": "W",
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "features": "women_hca_adj_qg_diff,seed_diff,close_win_pct_diff",
                "alpha_fallback": args.alpha_fallback,
                "alpha_step": args.alpha_step,
                "alpha_calibration_seasons": args.alpha_calibration_seasons,
                "inference_alpha": inference_alpha,
                "elo_initial_rating": args.elo_initial_rating,
                "elo_k_factor": args.elo_k_factor,
                "elo_home_advantage": args.elo_home_advantage,
                "elo_season_carryover": args.elo_season_carryover,
                "elo_scale": args.elo_scale,
                "elo_mov_alpha": args.elo_mov_alpha,
                "elo_weight_regular": args.elo_weight_regular,
                "elo_weight_tourney": args.elo_weight_tourney,
                "women_hca": args.women_hca,
            }
        )
        mlflow.log_metrics(
            {
                "flat_brier": summary.overall_flat_brier,
                "log_loss": summary.overall_log_loss,
                "selected_alpha_mean": float(summary.per_season_metrics["selected_alpha"].mean()),
            }
        )
        r1_values = summary.per_season_metrics["r1_brier"].dropna()
        if not r1_values.empty:
            mlflow.log_metric("r1_brier_mean", float(r1_values.mean()))
        r2_values = summary.per_season_metrics["r2plus_brier"].dropna()
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
