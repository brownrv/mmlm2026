from __future__ import annotations

import pandas as pd

from mmlm2026.analysis.benchmark_gap import (
    build_benchmark_gap_table,
    prioritize_gap_cells,
    summarize_gap_cells,
)


def test_benchmark_gap_analysis_summarizes_and_prioritizes_cells() -> None:
    local = pd.DataFrame(
        {
            "Season": [2024] * 9,
            "league": ["M"] * 9,
            "ID": [f"2024_1_{team_id}" for team_id in range(2, 11)],
            "LowTeamID": [1] * 9,
            "HighTeamID": list(range(2, 11)),
            "was_played": [True] * 9,
            "outcome": [1.0] * 8 + [0.0],
            "Pred": [0.70] * 8 + [0.20],
            "play_prob": [0.80] * 8 + [0.40],
            "bucket": ["very_likely"] * 8 + ["likely"],
            "actual_round": [2] * 9,
            "actual_round_group": ["R2+"] * 9,
            "predicted_round": [2] * 9,
            "predicted_round_group": ["R2+"] * 9,
            "brier_component": [0.09] * 8 + [0.04],
        }
    )
    reference = pd.DataFrame(
        {
            "Season": [2024] * 9,
            "league": ["M"] * 9,
            "ID": [f"2024_1_{team_id}" for team_id in range(2, 11)],
            "benchmark_pred": [0.80] * 8 + [0.10],
            "benchmark_round": [2] * 9,
            "benchmark_round_group": ["R2+"] * 9,
            "benchmark_bucket": ["very_likely"] * 8 + ["likely"],
            "benchmark_play_prob": [0.85] * 8 + [0.35],
            "benchmark_occurred": [True] * 9,
            "benchmark_actual_winner_id": [1] * 8 + [10],
        }
    )

    merged = build_benchmark_gap_table(local, reference)

    assert "pred_delta" in merged.columns
    assert "brier_gap" in merged.columns

    summary = summarize_gap_cells(merged)
    assert summary["played_games"].sum() == 9

    priority = prioritize_gap_cells(summary)
    assert len(priority) == 1
    assert priority.loc[0, "benchmark_bucket"] == "very_likely"
