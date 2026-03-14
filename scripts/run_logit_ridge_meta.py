from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]

from mmlm2026.evaluation.meta import (
    fit_logit_ridge_meta,
    logit_clip,
    merge_meta_prediction_frames,
    score_logit_ridge_meta,
    tune_logit_ridge_alpha,
)
from mmlm2026.utils.mlflow_tracking import start_tracked_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the logit-Ridge meta-learner challenger.")
    parser.add_argument("--league", choices=["M", "W"], required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/meta_runs"))
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--meta-train-seasons",
        nargs="+",
        type=int,
        default=[2018, 2019, 2021, 2022],
    )
    parser.add_argument("--holdout-seasons", nargs="+", type=int, default=[2023, 2024])
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = args.output_dir / f"{args.league.lower()}_logit_ridge_meta"
    output_dir.mkdir(parents=True, exist_ok=True)

    members = _member_specs(args.league)
    meta_train_frames = [
        _run_member(
            args,
            member_name=name,
            extra_args=extra_args,
            seasons=args.meta_train_seasons,
            phase="meta_train",
        )
        for name, extra_args in members
    ]
    holdout_frames = [
        _run_member(
            args,
            member_name=name,
            extra_args=extra_args,
            seasons=args.holdout_seasons,
            phase="holdout",
        )
        for name, extra_args in members
    ]

    pred_cols = [f"pred_{name}" for name, _ in members]
    meta_train = merge_meta_prediction_frames(meta_train_frames, pred_col_names=pred_cols)
    holdout = merge_meta_prediction_frames(holdout_frames, pred_col_names=pred_cols)

    for pred_col in pred_cols:
        meta_train[f"logit_{pred_col}"] = logit_clip(meta_train[pred_col])
        holdout[f"logit_{pred_col}"] = logit_clip(holdout[pred_col])
    feature_cols = [f"logit_{pred_col}" for pred_col in pred_cols]

    tuning = tune_logit_ridge_alpha(meta_train, feature_cols=feature_cols)
    model = fit_logit_ridge_meta(meta_train, feature_cols=feature_cols, alpha=tuning.alpha)
    holdout["pred"] = score_logit_ridge_meta(model, holdout, feature_cols=feature_cols)

    metrics = (
        holdout.groupby("Season", as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "flat_brier": float(brier_score_loss(group["outcome"], group["pred"])),
                    "log_loss": float(log_loss(group["outcome"], group["pred"])),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    overall_brier = float(brier_score_loss(holdout["outcome"], holdout["pred"]))
    overall_log_loss = float(log_loss(holdout["outcome"], holdout["pred"]))

    tuning.per_alpha.to_csv(output_dir / "alpha_grid.csv", index=False)
    metrics.to_csv(output_dir / "per_season_metrics.csv", index=False)
    holdout.to_csv(output_dir / "holdout_predictions.csv", index=False)

    if args.log_mlflow:
        with start_tracked_run(
            args.run_name,
            tags={
                "hypothesis": (
                    "A logit-Ridge meta-learner may beat the frozen leader "
                    "using current near-miss challengers."
                ),
                "model_family": "logit_ridge_meta",
                "league": "men" if args.league == "M" else "women",
                "depends_on": "late_arch_meta_01_v1",
                "retest_if": "base member set or holdout protocol changes",
                "leakage_audit": "passed",
            },
        ):
            mlflow.log_params(
                {
                    "league": args.league,
                    "members": ",".join(name for name, _ in members),
                    "meta_train_seasons": ",".join(
                        str(season) for season in args.meta_train_seasons
                    ),
                    "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                    "alpha": tuning.alpha,
                }
            )
            mlflow.log_metrics(
                {
                    "flat_brier": overall_brier,
                    "log_loss": overall_log_loss,
                }
            )
            mlflow.log_artifact(str(output_dir / "alpha_grid.csv"))
            mlflow.log_artifact(str(output_dir / "per_season_metrics.csv"))
            mlflow.log_artifact(str(output_dir / "holdout_predictions.csv"))

    print(metrics.to_string(index=False))
    print(f"\nOverall metrics: flat_brier={overall_brier:.6f} log_loss={overall_log_loss:.6f}")
    print(f"Best alpha: {tuning.alpha}")
    print(f"Artifacts written under {output_dir}")
    return 0


def _member_specs(league: str) -> list[tuple[str, list[str]]]:
    if league == "M":
        return [
            ("frozen", []),
            ("espn_strength", ["--include-ridge-strength", "--include-espn-four-factor"]),
            ("pythag", ["--include-pythag"]),
        ]
    return [
        ("frozen", ["--include-espn-four-factor"]),
        ("routed_basic", []),
        ("seed_gap", ["--include-espn-four-factor", "--include-seed-elo-gap"]),
    ]


def _run_member(
    args: argparse.Namespace,
    *,
    member_name: str,
    extra_args: list[str],
    seasons: list[int],
    phase: str,
) -> pd.DataFrame:
    run_output = args.output_dir / "meta_members" / args.league.lower() / phase / member_name
    run_output.mkdir(parents=True, exist_ok=True)
    if args.league == "M":
        script = Path("scripts/run_men_reference_margin.py")
        pred_path = run_output / "m_reference_margin" / "validation" / "oof_predictions.csv"
    else:
        script = Path("scripts/run_women_routed_round_group_model.py")
        pred_path = run_output / "w_routed_round_group" / "validation" / "oof_predictions.csv"

    if pred_path.exists():
        return pd.read_csv(pred_path)

    cmd = [
        sys.executable,
        str(script),
        "--holdout-seasons",
        *[str(season) for season in seasons],
        "--output-dir",
        str(run_output),
        "--run-name",
        f"meta-{args.league.lower()}-{phase}-{member_name}",
        *extra_args,
    ]
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    if args.league == "M":
        env["OMP_NUM_THREADS"] = "1"
        env["LOKY_MAX_CPU_COUNT"] = "1"
    subprocess.run(cmd, check=True, env=env)
    return pd.read_csv(pred_path)


if __name__ == "__main__":
    raise SystemExit(main())
