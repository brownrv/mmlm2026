from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts


def _build_slots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "Slot": ["R1A", "R1B", "R2A"],
            "StrongSeed": ["A01", "A02", "R1A"],
            "WeakSeed": ["A04", "A03", "R1B"],
        }
    )


def _build_seeds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Season": [2025, 2025, 2025, 2025],
            "Seed": ["A01", "A02", "A03", "A04"],
            "TeamID": [1, 2, 3, 4],
        }
    )


def _build_matchup_probs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [
                "2025_1_2",
                "2025_1_3",
                "2025_1_4",
                "2025_2_3",
                "2025_2_4",
                "2025_3_4",
            ],
            "Pred": [0.60, 0.65, 0.80, 0.70, 0.75, 0.55],
        }
    )


def _build_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "WTeamID": [1, 2, 1],
            "LTeamID": [4, 3, 2],
            "Round": [1, 1, 2],
        }
    )


def test_compute_bracket_diagnostics_returns_expected_play_probabilities() -> None:
    diagnostics = compute_bracket_diagnostics(
        _build_slots(),
        _build_seeds(),
        _build_matchup_probs(),
        season=2025,
    )

    play_prob = diagnostics.play_probabilities.set_index(["LowTeamID", "HighTeamID"])["play_prob"]

    assert play_prob[(1, 4)] == 1.0
    assert play_prob[(2, 3)] == 1.0
    assert play_prob[(1, 2)] == pytest.approx(0.56)
    assert play_prob[(1, 3)] == pytest.approx(0.24)
    assert play_prob[(2, 4)] == pytest.approx(0.14)
    assert play_prob[(3, 4)] == pytest.approx(0.06)


def test_compute_bracket_diagnostics_summarizes_realized_results() -> None:
    diagnostics = compute_bracket_diagnostics(
        _build_slots(),
        _build_seeds(),
        _build_matchup_probs(),
        season=2025,
        results=_build_results(),
    )

    assert diagnostics.bucket_summary is not None
    assert diagnostics.round_summary is not None
    assert set(diagnostics.round_summary["Round"]) == {1, 2}


def test_save_bracket_artifacts_writes_csvs(tmp_path: Path) -> None:
    diagnostics = compute_bracket_diagnostics(
        _build_slots(),
        _build_seeds(),
        _build_matchup_probs(),
        season=2025,
        results=_build_results(),
    )

    artifacts = save_bracket_artifacts(diagnostics, output_dir=tmp_path)

    assert artifacts.play_prob_path.exists()
    assert artifacts.slot_team_prob_path.exists()
    assert artifacts.bucket_summary_path is not None and artifacts.bucket_summary_path.exists()
    assert artifacts.round_summary_path is not None and artifacts.round_summary_path.exists()
