from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from mmlm2026.evaluation.validation import save_validation_artifacts, validate_season_holdouts
from mmlm2026.utils.mlflow_tracking import start_tracked_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run leave-season-out CV validation.")
    parser.add_argument("feature_table", type=Path, help="Path to a CSV or Parquet feature table.")
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Feature columns to pass to the model.",
    )
    parser.add_argument(
        "--holdout-seasons",
        nargs="+",
        type=int,
        required=True,
        help="Validation seasons to hold out.",
    )
    parser.add_argument("--season-col", default="Season")
    parser.add_argument("--outcome-col", default="outcome")
    parser.add_argument("--round-group-col", default="round_group")
    parser.add_argument("--train-min-games", type=int, default=50)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "interim" / "validation",
        help="Directory for generated validation artifacts.",
    )
    parser.add_argument("--run-name", default="validate-cv")
    parser.add_argument("--league", default="combined")
    parser.add_argument(
        "--hypothesis",
        default="leave-season-out validation run",
        help="Hypothesis tag for MLflow logging.",
    )
    parser.add_argument(
        "--depends-on",
        default="feature_table",
        help="Dependency tag for MLflow logging.",
    )
    parser.add_argument(
        "--retest-if",
        default="validation split policy changes",
        help="Retest condition tag for MLflow logging.",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log metrics, params, and artifacts to MLflow.",
    )
    parser.add_argument(
        "--leakage-audit",
        choices=["passed", "failed"],
        default=None,
        help="Leakage-audit tag required when --log-mlflow is used.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.log_mlflow and args.leakage_audit is None:
        parser.error("--log-mlflow requires --leakage-audit so the run records audit status.")

    frame = _load_feature_table(args.feature_table)
    summary = validate_season_holdouts(
        frame,
        feature_cols=args.features,
        holdout_seasons=args.holdout_seasons,
        season_col=args.season_col,
        outcome_col=args.outcome_col,
        round_group_col=args.round_group_col,
        train_min_games=args.train_min_games,
        calibration_bins=args.calibration_bins,
    )
    artifacts = save_validation_artifacts(summary, output_dir=args.output_dir)

    if args.log_mlflow:
        _log_mlflow_run(args, summary, artifacts)

    print(summary.per_season_metrics.to_string(index=False))
    print(
        "\nOverall metrics:",
        f"flat_brier={summary.overall_flat_brier:.6f}",
        f"log_loss={summary.overall_log_loss:.6f}",
    )
    print(f"Artifacts written to {args.output_dir}")
    return 0


def _load_feature_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported feature table format: {path.suffix}")


def _log_mlflow_run(
    args: argparse.Namespace,
    summary,
    artifacts,
) -> None:
    tags = {
        "hypothesis": args.hypothesis,
        "model_family": "logistic_regression",
        "league": args.league,
        "season_window": f"pre-{min(args.holdout_seasons)}",
        "depends_on": args.depends_on,
        "retest_if": args.retest_if,
        "leakage_audit": args.leakage_audit,
    }

    with start_tracked_run(args.run_name, tags=tags):
        mlflow.log_params(
            {
                "feature_table": str(args.feature_table),
                "features": ",".join(args.features),
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "train_min_games": args.train_min_games,
                "calibration_bins": args.calibration_bins,
            }
        )
        mlflow.log_metrics(
            {
                "flat_brier": summary.overall_flat_brier,
                "log_loss": summary.overall_log_loss,
            }
        )
        for _, row in summary.per_season_metrics.iterrows():
            season = int(row["Season"])
            mlflow.log_metric(f"flat_brier_season_{season}", float(row["flat_brier"]))
            if "r1_brier" in row and pd.notna(row["r1_brier"]):
                mlflow.log_metric(f"r1_brier_season_{season}", float(row["r1_brier"]))
            if "r2plus_brier" in row and pd.notna(row["r2plus_brier"]):
                mlflow.log_metric(f"r2plus_brier_season_{season}", float(row["r2plus_brier"]))

        mlflow.log_artifact(str(artifacts.per_season_metrics_path))
        mlflow.log_artifact(str(artifacts.calibration_table_path))
        mlflow.log_artifact(str(artifacts.oof_predictions_path))
        mlflow.log_artifact(str(artifacts.reliability_diagram_path))


if __name__ == "__main__":
    raise SystemExit(main())
