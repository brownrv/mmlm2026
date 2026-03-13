from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mmlm2026.analysis.benchmark_gap import (
    build_benchmark_gap_table,
    load_reference_predictions,
    prioritize_gap_cells,
    summarize_gap_cells,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare frozen all-matchups predictions against the reference benchmark."
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        default=Path("data/interim/frozen_historical_all_matchups.parquet"),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("data/reference"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/benchmark_gap_analysis"),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    local = pd.read_parquet(args.local_path)
    reference = load_reference_predictions(args.reference_root)
    merged = build_benchmark_gap_table(local, reference)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / "benchmark_gap_all_matchups.parquet"
    summary_path = args.output_dir / "benchmark_gap_round_bucket_summary.parquet"
    season_path = args.output_dir / "benchmark_gap_season_round_bucket_summary.parquet"
    worst_path = args.output_dir / "benchmark_gap_priority_cells.parquet"

    merged.to_parquet(merged_path, index=False)
    merged.to_csv(merged_path.with_suffix(".csv"), index=False)

    summary = summarize_gap_cells(merged)
    summary.to_parquet(summary_path, index=False)
    summary.to_csv(summary_path.with_suffix(".csv"), index=False)

    season_summary = summarize_gap_cells(merged, by_season=True)
    season_summary.to_parquet(season_path, index=False)
    season_summary.to_csv(season_path.with_suffix(".csv"), index=False)

    priority = prioritize_gap_cells(summary)
    priority.to_parquet(worst_path, index=False)
    priority.to_csv(worst_path.with_suffix(".csv"), index=False)

    print(f"Wrote merged comparison rows to {merged_path}")
    print(f"Wrote round-bucket summary to {summary_path}")
    print(f"Wrote season round-bucket summary to {season_path}")
    print(f"Wrote prioritized gap cells to {worst_path}")
    if not priority.empty:
        print("\nPriority cells:")
        print(priority.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
