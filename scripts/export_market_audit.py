from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mmlm2026.analysis.market_audit import (
    compare_market_to_local_tourney,
    load_betexplorer_frames,
    summarize_market_coverage,
    summarize_market_viability,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit BetExplorer coverage and leakage risk.")
    parser.add_argument(
        "--market-root",
        type=Path,
        default=Path("data/processed/betexplorer"),
    )
    parser.add_argument(
        "--local-played-path",
        type=Path,
        default=Path("data/interim/frozen_historical_performance.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/market_audit"),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    games = load_betexplorer_frames(args.market_root)
    local_played = pd.read_parquet(args.local_played_path)

    coverage = summarize_market_coverage(games)
    viability = summarize_market_viability(coverage)
    tourney_compare = compare_market_to_local_tourney(games, local_played)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(coverage, args.output_dir / "betexplorer_coverage_by_season.parquet")
    _write_frame(viability, args.output_dir / "betexplorer_viability_summary.parquet")
    _write_frame(tourney_compare, args.output_dir / "betexplorer_tourney_vs_local.parquet")

    print("BetExplorer viability summary:")
    print(viability.to_string(index=False))
    print("\nTournament comparison vs frozen leaders:")
    print(tourney_compare.to_string(index=False))
    return 0


def _write_frame(frame: pd.DataFrame, parquet_path: Path) -> None:
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(parquet_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    raise SystemExit(main())
