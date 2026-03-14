from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mmlm2026.analysis.espn_audit import (
    load_espn_feature_frames,
    summarize_espn_coverage,
    summarize_espn_null_profile,
    summarize_espn_viability,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit ESPN-derived feature stability by season.")
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
        default=Path("data/interim/espn_audit"),
    )
    parser.add_argument("--season-floor", type=int, default=2004)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    feature_frames, team_universe = load_espn_feature_frames(
        data_dir=args.data_dir,
        espn_root=args.espn_root,
        team_spellings_path=args.team_spellings_path,
        season_floor=args.season_floor,
    )
    coverage = summarize_espn_coverage(feature_frames, team_universe)
    null_profile = summarize_espn_null_profile(feature_frames)
    viability = summarize_espn_viability(coverage, null_profile)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(coverage, args.output_dir / "espn_coverage_by_season.parquet")
    _write_frame(null_profile, args.output_dir / "espn_null_profile.parquet")
    _write_frame(viability, args.output_dir / "espn_viability_summary.parquet")

    print("ESPN viability summary:")
    print(viability.to_string(index=False))
    return 0


def _write_frame(frame: pd.DataFrame, parquet_path: Path) -> None:
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(parquet_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    raise SystemExit(main())
