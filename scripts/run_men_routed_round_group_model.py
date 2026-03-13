from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from run_men_reference_margin import (
    _apply_temperature,
    _blend_probabilities,
    _drop_required_missing,
    _elo_probabilities,
    _estimate_residual_sigma,
    _fit_reference_calibration,
    _group_brier,
    _margin_to_probability,
    _mirror_margin_training_rows,
    _prepare_reference_feature_frame,
    build_margin_pipeline,
)
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts
from mmlm2026.evaluation.men_round_group import route_group_predictions
from mmlm2026.evaluation.validation import (
    ValidationSummary,
    build_calibration_table,
    save_validation_artifacts,
)
from mmlm2026.features.primary import build_phase_ab_submission_features
from mmlm2026.submission.frozen_models import (
    MEN_ELO_PARAMS,
    build_seeded_submission_rows,
    load_frozen_men_context,
)
from mmlm2026.utils.mlflow_tracking import start_tracked_run

FEATURE_COLS = [
    "adj_qg_diff",
    "mov_per100_diff",
    "seed_diff",
    "ast_rate_diff",
    "close_win_pct_diff",
    "conf_tourney_win_pct_diff",
    "ftr_diff",
    "tov_rate_diff",
]
CALIBRATION_FEATURE_COLS = [
    "adj_qg_diff",
    "mov_per100_diff",
    "seed_diff",
    "ast_rate_diff",
]
ROUND_GROUPS = ("R1", "R2+")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the routed men R1-vs-R2+ late challenger.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument("--holdout-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/challenger_runs"))
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--run-name", default="late-arch-rg-07-men-routed-round-group")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    context = load_frozen_men_context(args.data_dir)
    feature_table = context.training.copy()

    output_dir = args.output_dir / "m_routed_round_group"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_table.to_parquet(
            output_dir / "men_routed_round_group_features.parquet",
            index=False,
        )

    summary = _validate_holdouts(
        feature_table,
        holdout_seasons=args.holdout_seasons,
        seeds=context.seeds,
        team_features=context.team_features,
        elo_scale=MEN_ELO_PARAMS["scale"],
    )
    validation_artifacts = save_validation_artifacts(summary, output_dir=output_dir / "validation")

    latest_holdout = max(args.holdout_seasons)
    infer_frame = _prepare_reference_feature_frame(
        build_phase_ab_submission_features(
            build_seeded_submission_rows(context.seeds, season=latest_holdout)[
                ["Season", "LowTeamID", "HighTeamID"]
            ],
            context.seeds,
            context.team_features,
            league="M",
        )
    )
    infer_frame = _score_inference_frame(
        feature_table,
        infer_frame,
        season=latest_holdout,
        seeds=context.seeds,
        team_features=context.team_features,
        elo_scale=MEN_ELO_PARAMS["scale"],
    )
    infer_frame["ID"] = [
        f"{latest_holdout}_{low}_{high}"
        for low, high in zip(infer_frame["LowTeamID"], infer_frame["HighTeamID"], strict=False)
    ]

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
    seeds: pd.DataFrame,
    team_features: pd.DataFrame,
    elo_scale: float,
    train_min_games: int = 50,
) -> ValidationSummary:
    work = frame.loc[frame["outcome"].notna() & frame["margin"].notna()].copy()
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

        scored = _score_validation_frame(
            frame=frame,
            train_frame=train,
            valid_frame=valid,
            season=holdout_season,
            seeds=seeds,
            team_features=team_features,
            elo_scale=elo_scale,
        )
        prediction_frames.append(scored)
        metrics_rows.append(
            {
                "Season": holdout_season,
                "train_tourney_games": len(train),
                "valid_tourney_games": len(valid),
                "flat_brier": float(brier_score_loss(scored["outcome"], scored["pred"])),
                "log_loss": float(log_loss(scored["outcome"], scored["pred"])),
                "r1_brier": _group_brier(scored, "R1"),
                "r2plus_brier": _group_brier(scored, "R2+"),
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


def _score_validation_frame(
    *,
    frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    season: int,
    seeds: pd.DataFrame,
    team_features: pd.DataFrame,
    elo_scale: float,
) -> pd.DataFrame:
    routed_preds: dict[str, pd.Series] = {}
    fallback_model, fallback_sigma, fallback_calibration_state = _fit_group_bundle(
        frame=frame,
        train_frame=train_frame,
        round_group=None,
        season=season,
        seeds=seeds,
        team_features=team_features,
        elo_scale=elo_scale,
    )
    fallback_margin = pd.Series(
        fallback_model.predict(valid_frame[FEATURE_COLS]),
        index=valid_frame.index,
    )
    fallback_raw = _margin_to_probability(fallback_margin, fallback_sigma)
    fallback_cal = _apply_temperature(
        fallback_raw,
        fallback_calibration_state.temperature,
    )
    fallback_elo = _elo_probabilities(valid_frame, scale=elo_scale)
    fallback_pred = _blend_probabilities(
        fallback_cal,
        fallback_elo,
        fallback_calibration_state.alpha,
    )
    for round_group in ROUND_GROUPS:
        model, sigma, calibration_state = _fit_group_bundle(
            frame=frame,
            train_frame=train_frame,
            round_group=round_group,
            season=season,
            seeds=seeds,
            team_features=team_features,
            elo_scale=elo_scale,
        )
        group_valid = valid_frame.copy()
        pred_margin = pd.Series(model.predict(group_valid[FEATURE_COLS]), index=group_valid.index)
        p_raw = _margin_to_probability(pred_margin, sigma)
        p_cal = _apply_temperature(p_raw, calibration_state.temperature)
        p_elo = _elo_probabilities(group_valid, scale=elo_scale)
        routed_preds[round_group] = _blend_probabilities(p_cal, p_elo, calibration_state.alpha)

    pred = route_group_predictions(
        routed_preds,
        valid_frame["round_group"],
        fallback=fallback_pred,
    )
    return pd.DataFrame(
        {
            "Season": valid_frame["Season"].to_numpy(),
            "LowTeamID": valid_frame["LowTeamID"].to_numpy(),
            "HighTeamID": valid_frame["HighTeamID"].to_numpy(),
            "outcome": valid_frame["outcome"].astype(int).to_numpy(),
            "pred": pred.to_numpy(),
            "round_group": valid_frame["round_group"].to_numpy(),
            "Round": valid_frame["Round"].to_numpy(),
        }
    )


def _score_inference_frame(
    frame: pd.DataFrame,
    infer_frame: pd.DataFrame,
    *,
    season: int,
    seeds: pd.DataFrame,
    team_features: pd.DataFrame,
    elo_scale: float,
) -> pd.DataFrame:
    train_frame = _drop_required_missing(frame.loc[frame["Season"] < season].copy())
    routed_preds: dict[str, pd.Series] = {}
    fallback_model, fallback_sigma, fallback_calibration_state = _fit_group_bundle(
        frame=frame,
        train_frame=train_frame,
        round_group=None,
        season=season,
        seeds=seeds,
        team_features=team_features,
        elo_scale=elo_scale,
    )
    fallback_margin = pd.Series(
        fallback_model.predict(infer_frame[FEATURE_COLS]),
        index=infer_frame.index,
    )
    fallback_raw = _margin_to_probability(fallback_margin, fallback_sigma)
    fallback_cal = _apply_temperature(
        fallback_raw,
        fallback_calibration_state.temperature,
    )
    fallback_elo = _elo_probabilities(infer_frame, scale=elo_scale)
    fallback_pred = _blend_probabilities(
        fallback_cal,
        fallback_elo,
        fallback_calibration_state.alpha,
    )
    for round_group in ROUND_GROUPS:
        model, sigma, calibration_state = _fit_group_bundle(
            frame=frame,
            train_frame=train_frame,
            round_group=round_group,
            season=season,
            seeds=seeds,
            team_features=team_features,
            elo_scale=elo_scale,
        )
        pred_margin = pd.Series(model.predict(infer_frame[FEATURE_COLS]), index=infer_frame.index)
        p_raw = _margin_to_probability(pred_margin, sigma)
        p_cal = _apply_temperature(p_raw, calibration_state.temperature)
        p_elo = _elo_probabilities(infer_frame, scale=elo_scale)
        routed_preds[round_group] = _blend_probabilities(p_cal, p_elo, calibration_state.alpha)

    infer_frame = infer_frame.copy()
    infer_frame["Pred"] = route_group_predictions(
        routed_preds,
        infer_frame["round_group"],
        fallback=fallback_pred,
    )
    return infer_frame


def _fit_group_bundle(
    *,
    frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    round_group: str | None,
    season: int,
    seeds: pd.DataFrame,
    team_features: pd.DataFrame,
    elo_scale: float,
):
    if round_group is None:
        group_train = _drop_required_missing(train_frame.copy())
        calibration_frame = frame.copy()
    else:
        group_train = _drop_required_missing(
            train_frame.loc[train_frame["round_group"] == round_group].copy()
        )
        calibration_frame = frame.loc[frame["round_group"] == round_group].copy()
    if group_train.empty:
        raise ValueError(
            f"No training rows found for round_group={round_group} before season {season}."
        )
    model = build_margin_pipeline(FEATURE_COLS)
    mirrored_train = _mirror_margin_training_rows(group_train, FEATURE_COLS)
    model.fit(mirrored_train[FEATURE_COLS], mirrored_train["margin"])
    sigma = _estimate_residual_sigma(model, mirrored_train, FEATURE_COLS)
    calibration_state = _fit_reference_calibration(
        frame=calibration_frame,
        full_feature_cols=FEATURE_COLS,
        calibration_feature_cols=CALIBRATION_FEATURE_COLS,
        target_season=season,
        seeds=seeds,
        team_features=team_features,
        elo_scale=elo_scale,
    )
    return model, sigma, calibration_state


def _log_mlflow_run(
    args: argparse.Namespace,
    summary: ValidationSummary,
    validation_artifacts,
    bracket_artifacts,
) -> None:
    tags = {
        "hypothesis": (
            "A true routed men R1-vs-R2+ model may improve held-out flat Brier "
            "where round-group calibration alone did not"
        ),
        "model_family": "routed_margin_regression",
        "league": "men",
        "season_window": ",".join(str(season) for season in args.holdout_seasons),
        "depends_on": "frozen_men_reference_margin,later_round_diagnostics",
        "retest_if": "men routed round-group training definition changes",
        "leakage_audit": "passed",
    }
    with start_tracked_run(args.run_name, tags=tags):
        mlflow.log_params(
            {
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "features": ",".join(FEATURE_COLS),
                "round_groups": ",".join(ROUND_GROUPS),
                "elo_scale": MEN_ELO_PARAMS["scale"],
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
