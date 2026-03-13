from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from run_men_reference_margin import (
    _apply_temperature,
    _drop_required_missing,
    _elo_probabilities,
    _estimate_residual_sigma,
    _field_eff_sigma,
    _fit_reference_calibration,
    _group_brier,
    _margin_to_probability,
    _mirror_margin_training_rows,
    _optimize_temperature,
    _prepare_reference_feature_frame,
    build_margin_pipeline,
)
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts
from mmlm2026.evaluation.men_round_group import (
    RoundGroupCalibrationState,
    apply_round_group_blend,
)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the men round-group-specific calibration challenger."
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
    parser.add_argument("--run-name", default="val-06-men-round-group-calibration")
    parser.add_argument("--alpha-fallback", type=float, default=0.45)
    parser.add_argument("--cal-seasons", type=int, default=8)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    context = load_frozen_men_context(args.data_dir)
    feature_table = context.training.copy()

    output_dir = args.output_dir / "m_round_group_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_table.to_parquet(output_dir / "men_round_group_features.parquet", index=False)

    summary, final_state = _validate_holdouts(
        feature_table,
        holdout_seasons=args.holdout_seasons,
        seeds=context.seeds,
        team_features=context.team_features,
        elo_scale=MEN_ELO_PARAMS["scale"],
        alpha_fallback=args.alpha_fallback,
        cal_seasons=args.cal_seasons,
    )
    validation_artifacts = save_validation_artifacts(summary, output_dir=output_dir / "validation")

    latest_holdout = max(args.holdout_seasons)
    submission_rows = build_seeded_submission_rows(context.seeds, season=latest_holdout)
    infer_frame = build_phase_ab_submission_features(
        submission_rows[["Season", "LowTeamID", "HighTeamID"]],
        context.seeds,
        context.team_features,
        league="M",
    )
    infer_frame = _prepare_reference_feature_frame(infer_frame)
    train_frame = _drop_required_missing(
        feature_table.loc[feature_table["Season"] < latest_holdout]
    )
    model = build_margin_pipeline(FEATURE_COLS)
    mirrored_train = _mirror_margin_training_rows(train_frame, FEATURE_COLS)
    model.fit(mirrored_train[FEATURE_COLS], mirrored_train["margin"])
    sigma = _estimate_residual_sigma(model, mirrored_train, FEATURE_COLS)
    infer_margin = pd.Series(model.predict(infer_frame[FEATURE_COLS]), index=infer_frame.index)
    infer_frame["p_raw"] = _margin_to_probability(infer_margin, sigma)
    infer_frame["p_elo"] = _elo_probabilities(infer_frame, scale=MEN_ELO_PARAMS["scale"])
    infer_frame["Pred"] = apply_round_group_blend(
        infer_frame["p_raw"],
        infer_frame["p_elo"],
        infer_frame["round_group"],
        state=final_state,
        apply_temperature=_apply_temperature,
    )
    infer_frame["ID"] = submission_rows["ID"].to_numpy()

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
        (
            f"R1(T={final_state.temperature_by_group['R1']:.4f}, "
            f"alpha={final_state.alpha_by_group['R1']:.4f})"
        ),
        (
            f"R2+(T={final_state.temperature_by_group['R2+']:.4f}, "
            f"alpha={final_state.alpha_by_group['R2+']:.4f})"
        ),
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
    alpha_fallback: float,
    cal_seasons: int,
    train_min_games: int = 50,
) -> tuple[ValidationSummary, RoundGroupCalibrationState]:
    work = frame.loc[frame["outcome"].notna() & frame["margin"].notna()].copy()
    metrics_rows: list[dict[str, float | int | None]] = []
    prediction_frames: list[pd.DataFrame] = []
    final_state: RoundGroupCalibrationState | None = None

    for holdout_season in holdout_seasons:
        train = work.loc[work["Season"] < holdout_season].copy()
        valid = work.loc[(work["Season"] == holdout_season) & (work["Round"] > 0)].copy()
        if len(train) < train_min_games:
            raise ValueError(
                f"Training set for season {holdout_season} has {len(train)} games; "
                f"minimum required is {train_min_games}."
            )

        model = build_margin_pipeline(FEATURE_COLS)
        mirrored_train = _mirror_margin_training_rows(train, FEATURE_COLS)
        model.fit(mirrored_train[FEATURE_COLS], mirrored_train["margin"])
        sigma = _estimate_residual_sigma(model, mirrored_train, FEATURE_COLS)
        pred_margin = pd.Series(model.predict(valid[FEATURE_COLS]), index=valid.index)
        p_raw = _margin_to_probability(pred_margin, sigma)
        p_elo = _elo_probabilities(valid, scale=elo_scale)
        state = _fit_round_group_calibration(
            frame=frame,
            target_season=holdout_season,
            seeds=seeds,
            team_features=team_features,
            elo_scale=elo_scale,
            alpha_fallback=alpha_fallback,
            cal_seasons=cal_seasons,
        )
        pred = apply_round_group_blend(
            p_raw,
            p_elo,
            valid["round_group"],
            state=state,
            apply_temperature=_apply_temperature,
        )

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
                "r1_temperature": state.temperature_by_group["R1"],
                "r2plus_temperature": state.temperature_by_group["R2+"],
                "r1_alpha": state.alpha_by_group["R1"],
                "r2plus_alpha": state.alpha_by_group["R2+"],
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


def _fit_round_group_calibration(
    *,
    frame: pd.DataFrame,
    target_season: int,
    seeds: pd.DataFrame,
    team_features: pd.DataFrame,
    elo_scale: float,
    alpha_fallback: float,
    cal_seasons: int,
) -> RoundGroupCalibrationState:
    global_state = _fit_reference_calibration(
        frame=frame,
        full_feature_cols=FEATURE_COLS,
        calibration_feature_cols=CALIBRATION_FEATURE_COLS,
        target_season=target_season,
        seeds=seeds,
        team_features=team_features,
        elo_scale=elo_scale,
        cal_seasons=cal_seasons,
        alpha_fallback=alpha_fallback,
    )

    group_temperatures: dict[str, float] = {}
    group_alphas: dict[str, float] = {}
    for round_group in ("R1", "R2+"):
        state = _fit_group_specific_state(
            frame=frame,
            round_group=round_group,
            target_season=target_season,
            seeds=seeds,
            team_features=team_features,
            elo_scale=elo_scale,
            fallback=global_state,
            cal_seasons=cal_seasons,
            alpha_fallback=alpha_fallback,
        )
        group_temperatures[round_group] = state.temperature
        group_alphas[round_group] = state.alpha

    return RoundGroupCalibrationState(
        temperature_by_group=group_temperatures,
        alpha_by_group=group_alphas,
        fallback_temperature=global_state.temperature,
        fallback_alpha=global_state.alpha,
    )


def _fit_group_specific_state(
    *,
    frame: pd.DataFrame,
    round_group: str,
    target_season: int,
    seeds: pd.DataFrame,
    team_features: pd.DataFrame,
    elo_scale: float,
    fallback,
    cal_seasons: int,
    alpha_fallback: float,
):
    prior_seasons = sorted(
        int(season) for season in frame["Season"].unique() if int(season) < target_season
    )
    calibration_seasons = prior_seasons[-cal_seasons:]
    if not calibration_seasons:
        return fallback

    t_rows: list[dict[str, float]] = []
    pooled_probabilities: list[float] = []
    pooled_outcomes: list[int] = []
    alpha_values: list[float] = []
    alpha_grid = np.linspace(0.0, 1.0, 101)

    for season in calibration_seasons:
        train = frame.loc[frame["Season"] < season].copy()
        valid = frame.loc[
            (frame["Season"] == season)
            & (frame["Round"] > 0)
            & (frame["round_group"] == round_group)
        ].copy()
        train = _drop_required_missing(train)
        valid = _drop_required_missing(valid)
        if train.empty or len(valid) < 8:
            continue

        v4_model = build_margin_pipeline(CALIBRATION_FEATURE_COLS)
        mirrored_v4 = _mirror_margin_training_rows(train, CALIBRATION_FEATURE_COLS)
        v4_model.fit(mirrored_v4[CALIBRATION_FEATURE_COLS], mirrored_v4["margin"])
        sigma_v4 = _estimate_residual_sigma(v4_model, mirrored_v4, CALIBRATION_FEATURE_COLS)
        valid_margin = pd.Series(
            v4_model.predict(valid[CALIBRATION_FEATURE_COLS]),
            index=valid.index,
        )
        p_raw = _margin_to_probability(valid_margin, sigma_v4)
        optimal_t = _optimize_temperature(p_raw, valid["outcome"].astype(int))
        eff_sigma = _field_eff_sigma(season=season, seeds=seeds, team_features=team_features)
        t_rows.append({"eff_sigma": eff_sigma, "temperature": optimal_t})
        pooled_probabilities.extend(p_raw.tolist())
        pooled_outcomes.extend(valid["outcome"].astype(int).tolist())

        p_elo = _elo_probabilities(valid, scale=elo_scale)
        best_alpha = min(
            alpha_grid,
            key=lambda alpha: float(
                brier_score_loss(
                    valid["outcome"].astype(int),
                    (float(alpha) * p_elo + (1.0 - float(alpha)) * p_raw).clip(
                        lower=1e-10,
                        upper=1.0 - 1e-10,
                    ),
                )
            ),
        )
        alpha_values.append(float(best_alpha))

    if len(alpha_values) < 3 or not pooled_probabilities:
        return fallback

    pooled_t = _optimize_temperature(
        pd.Series(pooled_probabilities, dtype=float),
        pd.Series(pooled_outcomes, dtype=int),
    )
    if len(t_rows) >= 2:
        t_frame = pd.DataFrame(t_rows)
        coeffs = np.polyfit(t_frame["eff_sigma"], t_frame["temperature"], deg=1)
        target_eff_sigma = _field_eff_sigma(
            season=target_season,
            seeds=seeds,
            team_features=team_features,
        )
        dynamic_t = float(coeffs[0] * target_eff_sigma + coeffs[1])
        correlation = float(abs(t_frame["eff_sigma"].corr(t_frame["temperature"])))
        weight = min(max(correlation, 0.0), 1.0) if not np.isnan(correlation) else 0.0
        final_t = weight * dynamic_t + (1.0 - weight) * pooled_t
    else:
        final_t = pooled_t

    final_t = float(np.clip(final_t, 0.5, 3.0))
    final_alpha = float(np.mean(alpha_values)) if len(alpha_values) >= 5 else alpha_fallback
    return type(fallback)(temperature=final_t, alpha=final_alpha)


def _log_mlflow_run(
    args: argparse.Namespace,
    summary: ValidationSummary,
    validation_artifacts,
    bracket_artifacts,
    final_state: RoundGroupCalibrationState,
) -> None:
    tags = {
        "hypothesis": "Men round-group-specific calibration may reduce the persistent R2+ weakness",
        "model_family": "frozen_men_reference_margin_plus_round_group_calibration",
        "league": "men",
        "season_window": ",".join(str(season) for season in args.holdout_seasons),
        "depends_on": "frozen_men_reference_margin",
        "retest_if": "men round-group historical diagnostics change materially",
        "leakage_audit": "passed",
    }
    with start_tracked_run(args.run_name, tags=tags):
        mlflow.log_params(
            {
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "alpha_fallback": args.alpha_fallback,
                "cal_seasons": args.cal_seasons,
                "r1_temperature": final_state.temperature_by_group["R1"],
                "r2plus_temperature": final_state.temperature_by_group["R2+"],
                "r1_alpha": final_state.alpha_by_group["R1"],
                "r2plus_alpha": final_state.alpha_by_group["R2+"],
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
