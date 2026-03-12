from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mmlm2026.features.elo_tuning import (
    broad_search_ranges,
    default_include_playins,
    default_objective_seasons,
    default_report_seasons,
    narrow_ranges,
    prepare_game_rows,
    run_study,
    save_study_artifacts,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune men Elo parameters with persistent Optuna TPE."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument(
        "--mode",
        choices=["replication", "generalization"],
        default="replication",
    )
    parser.add_argument("--objective-seasons", nargs="+", type=int, default=None)
    parser.add_argument("--report-seasons", nargs="+", type=int, default=None)
    parser.add_argument(
        "--include-playins",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials-broad", type=int, default=500)
    parser.add_argument("--n-trials-narrow", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/tuning_runs/elo"))
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    objective_seasons = args.objective_seasons or default_objective_seasons(args.mode)
    report_seasons = args.report_seasons or default_report_seasons(args.mode)
    include_playins = (
        args.include_playins
        if args.include_playins is not None
        else default_include_playins(args.mode)
    )

    regular = pd.read_csv(args.data_dir / "MRegularSeasonCompactResults.csv")
    tourney = pd.read_csv(args.data_dir / "MNCAATourneyCompactResults.csv")
    seeds = pd.read_csv(args.data_dir / "MNCAATourneySeeds.csv")
    games = prepare_game_rows(
        regular=regular,
        tourney=tourney,
        seeds=seeds,
        include_playins=include_playins,
    )

    output_dir = args.output_dir / f"men_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_path = output_dir / "studies.sqlite3"
    broad_ranges = broad_search_ranges()
    broad_best = run_study(
        study_name=f"men_{args.mode}_broad",
        storage_path=storage_path,
        games=games,
        objective_seasons=objective_seasons,
        seed=args.seed,
        n_trials=args.n_trials_broad,
        ranges=broad_ranges,
        show_progress_bar=args.progress,
    )
    broad_artifacts = save_study_artifacts(
        output_dir=output_dir,
        study_name=f"men_{args.mode}_broad",
        storage_path=storage_path,
        best_params=broad_best,
        ranges=broad_ranges,
        objective_seasons=objective_seasons,
        report_seasons=report_seasons,
        include_playins=include_playins,
        mode=args.mode,
    )
    narrow_best = run_study(
        study_name=f"men_{args.mode}_narrow",
        storage_path=storage_path,
        games=games,
        objective_seasons=objective_seasons,
        seed=args.seed,
        n_trials=args.n_trials_narrow,
        ranges=narrow_ranges(broad_best, base_ranges=broad_ranges),
        show_progress_bar=args.progress,
    )
    narrow_artifacts = save_study_artifacts(
        output_dir=output_dir,
        study_name=f"men_{args.mode}_narrow",
        storage_path=storage_path,
        best_params=narrow_best,
        ranges=narrow_ranges(broad_best, base_ranges=broad_ranges),
        objective_seasons=objective_seasons,
        report_seasons=report_seasons,
        include_playins=include_playins,
        mode=args.mode,
    )

    print("Broad best:", broad_artifacts.best_params)
    print("Broad boundary hits:", broad_artifacts.boundary_hits)
    print("Narrow best:", narrow_artifacts.best_params)
    print("Narrow boundary hits:", narrow_artifacts.boundary_hits)
    print(f"Artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
