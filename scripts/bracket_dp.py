from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute bracket play probabilities and diagnostics."
    )
    parser.add_argument("season", type=int, help="Tournament season to analyze.")
    parser.add_argument(
        "--slots",
        type=Path,
        required=True,
        help="Path to NCAATourneySlots CSV.",
    )
    parser.add_argument(
        "--seeds",
        type=Path,
        required=True,
        help="Path to NCAATourneySeeds CSV.",
    )
    parser.add_argument(
        "--matchup-probs",
        type=Path,
        required=True,
        help="Path to matchup probabilities CSV with ID and Pred columns.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Optional tournament results CSV for realized-game diagnostics.",
    )
    parser.add_argument(
        "--round-col",
        default="Round",
        help="Round column name in the optional results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "interim" / "bracket_diagnostics",
        help="Directory to write diagnostic artifacts.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    slots = pd.read_csv(args.slots)
    seeds = pd.read_csv(args.seeds)
    matchup_probs = pd.read_csv(args.matchup_probs)
    results = pd.read_csv(args.results) if args.results is not None else None

    diagnostics = compute_bracket_diagnostics(
        slots,
        seeds,
        matchup_probs,
        season=args.season,
        results=results,
        round_col=args.round_col,
    )
    artifacts = save_bracket_artifacts(diagnostics, output_dir=args.output_dir)

    print(
        f"Computed {len(diagnostics.play_probabilities)} pair play probabilities "
        f"and {len(diagnostics.slot_team_probabilities)} slot-team probabilities."
    )
    print(f"Play probabilities written to {artifacts.play_prob_path}")
    if artifacts.bucket_summary_path is not None:
        print(f"Bucket summary written to {artifacts.bucket_summary_path}")
    if artifacts.round_summary_path is not None:
        print(f"Round summary written to {artifacts.round_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
