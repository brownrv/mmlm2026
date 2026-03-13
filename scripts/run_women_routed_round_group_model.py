from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts
from mmlm2026.evaluation.men_round_group import route_group_predictions
from mmlm2026.evaluation.validation import (
    ValidationSummary,
    build_calibration_table,
    build_logistic_pipeline,
    save_validation_artifacts,
)
from mmlm2026.features.elo import build_elo_seed_submission_features
from mmlm2026.submission.frozen_models import (
    build_seeded_submission_rows,
    load_frozen_women_context,
)
from mmlm2026.utils.mlflow_tracking import start_tracked_run

FEATURE_COLS = ["seed_diff", "elo_diff"]
ROUND_GROUPS = ("R1", "R2+")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the routed women R1-vs-R2+ late challenger.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument("--holdout-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/challenger_runs"))
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--run-name", default="late-arch-rg-08-women-routed-round-group")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    context = load_frozen_women_context(args.data_dir)
    feature_table = context.training.copy()

    output_dir = args.output_dir / "w_routed_round_group"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_table.to_parquet(
            output_dir / "women_routed_round_group_features.parquet",
            index=False,
        )

    summary = _validate_holdouts(feature_table, holdout_seasons=args.holdout_seasons)
    validation_artifacts = save_validation_artifacts(summary, output_dir=output_dir / "validation")

    latest_holdout = max(args.holdout_seasons)
    infer_rows = build_seeded_submission_rows(context.seeds, season=latest_holdout)
    infer_frame = build_elo_seed_submission_features(
        infer_rows[["Season", "LowTeamID", "HighTeamID"]],
        context.seeds,
        context.elo_ratings,
        league="W",
    )
    infer_frame = _score_inference_frame(feature_table, infer_frame, season=latest_holdout)
    infer_frame["ID"] = infer_rows["ID"].to_numpy()

    bracket = compute_bracket_diagnostics(
        context.slots,
        context.seeds,
        infer_frame[["ID", "Pred"]],
        season=latest_holdout,
        results=context.results,
        round_col="Round",
    )
    bracket_artifacts = save_bracket_artifacts(bracket, output_dir=output_dir / "bracket")

    if args.log_mlflow:
        _log_mlflow_run(args, summary, validation_artifacts, bracket_artifacts)

    print(summary.per_season_metrics.to_string(index=False))
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
    holdout_seasons: list[int],
    train_min_games: int = 50,
) -> ValidationSummary:
    work = frame.loc[frame["outcome"].notna()].copy()
    metrics_rows: list[dict[str, float | int | None]] = []
    prediction_frames: list[pd.DataFrame] = []

    for holdout_season in holdout_seasons:
        train = work.loc[work["Season"] < holdout_season].copy()
        valid = work.loc[(work["Season"] == holdout_season) & (work["Round"] > 0)].copy()
        if len(train) < train_min_games:
            raise ValueError(
                f"Training set for season {holdout_season} has {len(train)} games; "
                f"minimum required is {train_min_games}."
            )

        pred = _score_with_routed_models(train, valid, season=holdout_season)
        pred_frame = pd.DataFrame(
            {
                "Season": valid["Season"].to_numpy(),
                "LowTeamID": valid["LowTeamID"].to_numpy(),
                "HighTeamID": valid["HighTeamID"].to_numpy(),
                "outcome": valid["outcome"].astype(int).to_numpy(),
                "pred": pred.to_numpy(),
                "round_group": valid["round_group"].to_numpy(),
                "Round": valid["Round"].to_numpy(),
            }
        )
        prediction_frames.append(pred_frame)
        metrics_rows.append(
            {
                "Season": holdout_season,
                "train_tourney_games": len(train),
                "valid_tourney_games": len(valid),
                "flat_brier": float(brier_score_loss(pred_frame["outcome"], pred_frame["pred"])),
                "log_loss": float(log_loss(pred_frame["outcome"], pred_frame["pred"])),
                "r1_brier": _group_brier(pred_frame, "R1"),
                "r2plus_brier": _group_brier(pred_frame, "R2+"),
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


def _score_inference_frame(
    frame: pd.DataFrame,
    infer_frame: pd.DataFrame,
    *,
    season: int,
) -> pd.DataFrame:
    train = frame.loc[frame["Season"] < season].copy()
    pred = _score_with_routed_models(train, infer_frame, season=season)
    scored = infer_frame.copy()
    scored["Pred"] = pred
    return scored


def _score_with_routed_models(
    train_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    season: int,
) -> pd.Series:
    routed_preds: dict[str, pd.Series] = {}
    fallback_model = build_logistic_pipeline(FEATURE_COLS)
    fallback_model.fit(train_frame[FEATURE_COLS], train_frame["outcome"].astype(int))
    fallback_pred = pd.Series(
        fallback_model.predict_proba(target_frame[FEATURE_COLS])[:, 1],
        index=target_frame.index,
        dtype=float,
    )

    for round_group in ROUND_GROUPS:
        group_train = train_frame.loc[train_frame["round_group"] == round_group].copy()
        if group_train.empty:
            continue
        model = build_logistic_pipeline(FEATURE_COLS)
        model.fit(group_train[FEATURE_COLS], group_train["outcome"].astype(int))
        routed_preds[round_group] = pd.Series(
            model.predict_proba(target_frame[FEATURE_COLS])[:, 1],
            index=target_frame.index,
            dtype=float,
        )

    if not routed_preds:
        raise ValueError(f"No routed women training rows available before season {season}.")

    return route_group_predictions(
        routed_preds,
        target_frame["round_group"],
        fallback=fallback_pred,
    )


def _group_brier(frame: pd.DataFrame, round_group: str) -> float | None:
    mask = frame["round_group"] == round_group
    if not mask.any():
        return None
    return float(brier_score_loss(frame.loc[mask, "outcome"], frame.loc[mask, "pred"]))


def _log_mlflow_run(
    args: argparse.Namespace,
    summary: ValidationSummary,
    validation_artifacts,
    bracket_artifacts,
) -> None:
    tags = {
        "hypothesis": (
            "A true routed women R1-vs-R2+ model may improve held-out flat Brier "
            "over the unified ARCH-04B baseline"
        ),
        "model_family": "routed_logistic_regression",
        "league": "women",
        "season_window": ",".join(str(season) for season in args.holdout_seasons),
        "depends_on": "arch-04b,women_round_group_diagnostics",
        "retest_if": "women routed round-group training definition changes",
        "leakage_audit": "passed",
    }
    with start_tracked_run(args.run_name, tags=tags):
        mlflow.log_params(
            {
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "features": ",".join(FEATURE_COLS),
                "round_groups": ",".join(ROUND_GROUPS),
            }
        )
        mlflow.log_metrics(
            {
                "flat_brier": summary.overall_flat_brier,
                "log_loss": summary.overall_log_loss,
                "r1_brier_mean": float(summary.per_season_metrics["r1_brier"].dropna().mean()),
                "r2plus_brier_mean": float(
                    summary.per_season_metrics["r2plus_brier"].dropna().mean()
                ),
            }
        )
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
