from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mmlm2026.analysis.revised_detailed_audit import (
    build_refresh_dependency_matrix,
    compare_kaggle_vs_espn_discrepancies,
    load_revised_detailed_file_inventory,
    summarize_revised_vs_original,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit revised 2021-2026 detailed Kaggle files against originals."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument(
        "--espn-root",
        type=Path,
        default=Path("data/processed/espn"),
    )
    parser.add_argument(
        "--team-spellings-path",
        type=Path,
        default=Path("data/TeamSpellings.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/revised_detailed_audit"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/manifests/revised_detailed_2021_2026_inventory.csv"),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    inventory = load_revised_detailed_file_inventory(args.data_dir)
    _write_frame(inventory, args.output_dir / "revised_detailed_file_inventory.parquet")
    inventory.to_csv(args.manifest_path, index=False)

    men_regular_original = pd.read_csv(args.data_dir / "MRegularSeasonDetailedResults.csv")
    men_regular_revised = pd.read_csv(
        args.data_dir / "MRegularSeasonDetailedResults_2021_2026.csv"
    )
    women_regular_original = pd.read_csv(args.data_dir / "WRegularSeasonDetailedResults.csv")
    women_regular_revised = pd.read_csv(
        args.data_dir / "WRegularSeasonDetailedResults_2021_2026.csv"
    )
    men_tourney_original = pd.read_csv(args.data_dir / "MNCAATourneyDetailedResults.csv")
    men_tourney_revised = pd.read_csv(args.data_dir / "MNCAATourneyDetailedResults_2021_2026.csv")
    women_tourney_original = pd.read_csv(args.data_dir / "WNCAATourneyDetailedResults.csv")
    women_tourney_revised = pd.read_csv(args.data_dir / "WNCAATourneyDetailedResults_2021_2026.csv")

    season_frames = []
    stat_frames = []
    for league, scope, original, revised in [
        ("M", "regular", men_regular_original, men_regular_revised),
        ("W", "regular", women_regular_original, women_regular_revised),
        ("M", "tourney", men_tourney_original, men_tourney_revised),
        ("W", "tourney", women_tourney_original, women_tourney_revised),
    ]:
        season_summary, stat_summary = summarize_revised_vs_original(
            original=original,
            revised=revised,
            league=league,
            scope=scope,
        )
        season_frames.append(season_summary)
        stat_frames.append(stat_summary)

    season_summary = pd.concat(season_frames, ignore_index=True)
    stat_summary = pd.concat(stat_frames, ignore_index=True)
    dependency_matrix = build_refresh_dependency_matrix()

    _write_frame(season_summary, args.output_dir / "revised_detailed_season_summary.parquet")
    _write_frame(stat_summary, args.output_dir / "revised_detailed_stat_summary.parquet")
    _write_frame(dependency_matrix, args.output_dir / "revised_detailed_dependency_matrix.parquet")

    espn_summary = pd.concat(
        [
            compare_kaggle_vs_espn_discrepancies(
                league="M",
                regular_original=men_regular_original,
                regular_revised=men_regular_revised,
                tourney_original=men_tourney_original,
                tourney_revised=men_tourney_revised,
                espn_root=args.espn_root,
                team_spellings_path=args.team_spellings_path,
            ),
            compare_kaggle_vs_espn_discrepancies(
                league="W",
                regular_original=women_regular_original,
                regular_revised=women_regular_revised,
                tourney_original=women_tourney_original,
                tourney_revised=women_tourney_revised,
                espn_root=args.espn_root,
                team_spellings_path=args.team_spellings_path,
            ),
        ],
        ignore_index=True,
    )
    _write_frame(
        espn_summary,
        args.output_dir / "revised_detailed_espn_discrepancy_summary.parquet",
    )

    print("Revised detailed season summary:")
    print(season_summary.to_string(index=False))
    print("\nESPN discrepancy summary:")
    print(espn_summary.to_string(index=False))
    return 0


def _write_frame(frame: pd.DataFrame, parquet_path: Path) -> None:
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(parquet_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    raise SystemExit(main())
