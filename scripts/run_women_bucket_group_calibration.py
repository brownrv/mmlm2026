from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts
from mmlm2026.evaluation.validation import (
    ValidationSummary,
    build_calibration_table,
    build_logistic_pipeline,
    save_validation_artifacts,
)
from mmlm2026.evaluation.women_bucket_group import (
    WomenBucketCalibrationState,
    apply_bucket_group_temperature,
    derive_focus_group,
    optimize_temperature,
)
from mmlm2026.submission.frozen_models import (
    build_seeded_submission_rows,
    load_frozen_women_context,
)
from mmlm2026.utils.mlflow_tracking import start_tracked_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the women very_likely / R2+ calibration challenger."
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
    parser.add_argument("--run-name", default="val-07-women-bucket-group-calibration")
    parser.add_argument("--cal-seasons", type=int, default=8)
    parser.add_argument("--min-group-games", type=int, default=25)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    context = load_frozen_women_context(args.data_dir)
    feature_table = context.training.copy()

    output_dir = args.output_dir / "w_bucket_group_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_table.to_parquet(output_dir / "women_bucket_group_features.parquet", index=False)

    summary, final_state = _validate_holdouts(
        feature_table,
        holdout_seasons=args.holdout_seasons,
        seeds=context.seeds,
        slots=context.slots,
        results=context.results,
        elo_ratings=context.elo_ratings,
        cal_seasons=args.cal_seasons,
        min_group_games=args.min_group_games,
    )
    validation_artifacts = save_validation_artifacts(summary, output_dir=output_dir / "validation")

    latest_holdout = max(args.holdout_seasons)
    infer_frame, _ = _build_all_matchups(
        season=latest_holdout,
        train_frame=feature_table.loc[feature_table["Season"] < latest_holdout].copy(),
        seeds=context.seeds,
        slots=context.slots,
        results=context.results,
        elo_ratings=context.elo_ratings,
    )
    infer_frame["Pred"] = apply_bucket_group_temperature(
        infer_frame["Pred_base"],
        infer_frame["focus_group"],
        state=final_state,
    )
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
        _log_mlflow_run(
            args,
            summary,
            validation_artifacts,
            bracket_artifacts,
            final_state,
        )

    print(summary.per_season_metrics.to_string(index=False))
    print(
        "\nOverall metrics:",
        f"flat_brier={summary.overall_flat_brier:.6f}",
        f"log_loss={summary.overall_log_loss:.6f}",
    )
    print(
        "Final calibration:",
        f"baseline(T={final_state.fallback_temperature:.4f})",
        f"very_likely(T={final_state.temperature_by_group['very_likely']:.4f})",
        f"R2+(T={final_state.temperature_by_group['R2+']:.4f})",
    )
    print(f"Artifacts written under {output_dir}")
    return 0


def _validate_holdouts(
    frame: pd.DataFrame,
    *,
    holdout_seasons: list[int],
    seeds: pd.DataFrame,
    slots: pd.DataFrame,
    results: pd.DataFrame,
    elo_ratings: pd.DataFrame,
    cal_seasons: int,
    min_group_games: int,
    train_min_games: int = 50,
) -> tuple[ValidationSummary, WomenBucketCalibrationState]:
    work = frame.loc[frame["outcome"].notna()].copy()
    metrics_rows: list[dict[str, float | int | None]] = []
    prediction_frames: list[pd.DataFrame] = []
    final_state: WomenBucketCalibrationState | None = None

    for holdout_season in holdout_seasons:
        train = work.loc[work["Season"] < holdout_season].copy()
        if len(train) < train_min_games:
            raise ValueError(
                f"Training set for season {holdout_season} has {len(train)} games; "
                f"minimum required is {train_min_games}."
            )

        all_matchups, _ = _build_all_matchups(
            season=holdout_season,
            train_frame=train,
            seeds=seeds,
            slots=slots,
            results=results,
            elo_ratings=elo_ratings,
        )
        valid = work.loc[(work["Season"] == holdout_season) & (work["Round"] > 0)].copy()
        valid = valid.merge(
            all_matchups[
                [
                    "LowTeamID",
                    "HighTeamID",
                    "Pred_base",
                    "bucket",
                    "focus_group",
                ]
            ],
            on=["LowTeamID", "HighTeamID"],
            how="left",
            validate="one_to_one",
        )

        state = _fit_bucket_group_state(
            frame=frame,
            target_season=holdout_season,
            seeds=seeds,
            slots=slots,
            results=results,
            elo_ratings=elo_ratings,
            cal_seasons=cal_seasons,
            min_group_games=min_group_games,
        )
        pred = apply_bucket_group_temperature(
            valid["Pred_base"],
            valid["focus_group"],
            state=state,
        )

        pred_frame = pd.DataFrame(
            {
                "Season": valid["Season"].to_numpy(),
                "LowTeamID": valid["LowTeamID"].to_numpy(),
                "HighTeamID": valid["HighTeamID"].to_numpy(),
                "outcome": valid["outcome"].astype(int).to_numpy(),
                "pred": pred.to_numpy(),
                "round_group": valid["round_group"].to_numpy(),
                "bucket": valid["bucket"].to_numpy(),
                "focus_group": valid["focus_group"].to_numpy(),
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
                "r1_brier": _group_brier(pred_frame, "round_group", "R1"),
                "r2plus_brier": _group_brier(pred_frame, "round_group", "R2+"),
                "very_likely_brier": _group_brier(pred_frame, "bucket", "very_likely"),
                "baseline_temperature": state.fallback_temperature,
                "very_likely_temperature": state.temperature_by_group["very_likely"],
                "r2plus_temperature": state.temperature_by_group["R2+"],
            }
        )
        final_state = state

    if final_state is None:
        raise ValueError("No holdout seasons were evaluated.")

    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    calibration_table = build_calibration_table(
        oof_predictions["outcome"],
        oof_predictions["pred"],
        n_bins=10,
    )
    summary = ValidationSummary(
        per_season_metrics=pd.DataFrame(metrics_rows).sort_values("Season").reset_index(drop=True),
        calibration_table=calibration_table,
        oof_predictions=oof_predictions,
    )
    return summary, final_state


def _fit_bucket_group_state(
    frame: pd.DataFrame,
    *,
    target_season: int,
    seeds: pd.DataFrame,
    slots: pd.DataFrame,
    results: pd.DataFrame,
    elo_ratings: pd.DataFrame,
    cal_seasons: int,
    min_group_games: int,
) -> WomenBucketCalibrationState:
    prior_seasons = sorted(
        int(season) for season in frame["Season"].unique() if int(season) < target_season
    )
    calibration_seasons = prior_seasons[-cal_seasons:]
    pooled_probs: list[float] = []
    pooled_outcomes: list[int] = []
    group_probs: dict[str, list[float]] = {"very_likely": [], "R2+": []}
    group_outcomes: dict[str, list[int]] = {"very_likely": [], "R2+": []}

    for season in calibration_seasons:
        train = frame.loc[frame["Season"] < season].copy()
        if train.empty:
            continue
        all_matchups, _ = _build_all_matchups(
            season=season,
            train_frame=train,
            seeds=seeds,
            slots=slots,
            results=results,
            elo_ratings=elo_ratings,
        )
        valid = frame.loc[(frame["Season"] == season) & (frame["Round"] > 0)].copy()
        valid = valid.merge(
            all_matchups[
                [
                    "LowTeamID",
                    "HighTeamID",
                    "Pred_base",
                    "bucket",
                    "focus_group",
                ]
            ],
            on=["LowTeamID", "HighTeamID"],
            how="left",
            validate="one_to_one",
        )
        if valid.empty:
            continue
        pooled_probs.extend(float(value) for value in valid["Pred_base"])
        pooled_outcomes.extend(int(value) for value in valid["outcome"])
        for group_name in ("very_likely", "R2+"):
            group = valid.loc[valid["focus_group"] == group_name]
            if group.empty:
                continue
            group_probs[group_name].extend(float(value) for value in group["Pred_base"])
            group_outcomes[group_name].extend(int(value) for value in group["outcome"])

    if not pooled_outcomes:
        return WomenBucketCalibrationState(
            temperature_by_group={"very_likely": 1.0, "R2+": 1.0},
            fallback_temperature=1.0,
        )

    fallback_temperature = optimize_temperature(pooled_probs, pooled_outcomes)
    group_temperatures: dict[str, float] = {}
    for group_name in ("very_likely", "R2+"):
        if len(group_outcomes[group_name]) >= min_group_games:
            group_temperatures[group_name] = optimize_temperature(
                group_probs[group_name],
                group_outcomes[group_name],
            )
        else:
            group_temperatures[group_name] = fallback_temperature

    return WomenBucketCalibrationState(
        temperature_by_group=group_temperatures,
        fallback_temperature=fallback_temperature,
    )


def _build_all_matchups(
    *,
    season: int,
    train_frame: pd.DataFrame,
    seeds: pd.DataFrame,
    slots: pd.DataFrame,
    results: pd.DataFrame,
    elo_ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, object]:
    submission_rows = build_seeded_submission_rows(seeds, season=season)
    from mmlm2026.features.elo import build_elo_seed_submission_features

    infer_frame = build_elo_seed_submission_features(
        submission_rows[["Season", "LowTeamID", "HighTeamID"]],
        seeds,
        elo_ratings,
        league="W",
    )
    model = build_logistic_pipeline(["seed_diff", "elo_diff"])
    model.fit(train_frame[["seed_diff", "elo_diff"]], train_frame["outcome"].astype(int))
    infer_frame["Pred_base"] = model.predict_proba(infer_frame[["seed_diff", "elo_diff"]])[:, 1]
    infer_frame["ID"] = submission_rows["ID"].to_numpy()
    bracket = compute_bracket_diagnostics(
        slots,
        seeds,
        infer_frame[["ID", "Pred_base"]].rename(columns={"Pred_base": "Pred"}),
        season=season,
        results=results,
        round_col="Round",
    )
    infer_frame = infer_frame.merge(
        bracket.play_probabilities,
        on=["Season", "LowTeamID", "HighTeamID"],
        how="left",
        validate="one_to_one",
    )
    infer_frame["bucket"] = infer_frame["play_prob"].map(_bucket_from_play_prob)
    infer_frame["focus_group"] = derive_focus_group(
        infer_frame["round_group"],
        infer_frame["bucket"],
    )
    return infer_frame, bracket


def _bucket_from_play_prob(play_prob: float) -> str:
    if play_prob >= 0.99:
        return "definite"
    if play_prob >= 0.30:
        return "very_likely"
    if play_prob >= 0.10:
        return "likely"
    if play_prob >= 0.03:
        return "plausible"
    return "remote"


def _group_brier(frame: pd.DataFrame, group_col: str, group_name: str) -> float | None:
    mask = frame[group_col] == group_name
    if not mask.any():
        return None
    return float(brier_score_loss(frame.loc[mask, "outcome"], frame.loc[mask, "pred"]))


def _log_mlflow_run(
    args: argparse.Namespace,
    summary: ValidationSummary,
    validation_artifacts,
    bracket_artifacts,
    final_state: WomenBucketCalibrationState,
) -> None:
    tags = {
        "hypothesis": (
            "Women very_likely and R2+ games may benefit from targeted temperature "
            "calibration on top of the frozen women model"
        ),
        "model_family": "frozen_women_arch04b_plus_bucket_group_temperature",
        "league": "women",
        "season_window": ",".join(str(season) for season in args.holdout_seasons),
        "depends_on": "arch-04b",
        "retest_if": "women bucket diagnostics move materially after new data",
        "leakage_audit": "passed",
    }
    with start_tracked_run(args.run_name, tags=tags):
        mlflow.log_params(
            {
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "cal_seasons": args.cal_seasons,
                "min_group_games": args.min_group_games,
                "baseline_temperature": final_state.fallback_temperature,
                "very_likely_temperature": final_state.temperature_by_group["very_likely"],
                "r2plus_temperature": final_state.temperature_by_group["R2+"],
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
                "very_likely_brier_mean": float(
                    summary.per_season_metrics["very_likely_brier"].dropna().mean()
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
