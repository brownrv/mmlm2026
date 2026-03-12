from __future__ import annotations

import pandas as pd
import pytest

from mmlm2026.evaluation.ensemble import (
    blend_predictions,
    find_best_blend_weight,
    merge_prediction_frames,
)


def test_blend_predictions_returns_convex_average() -> None:
    blended = blend_predictions([0.2, 0.8], [0.6, 0.4], first_weight=0.75)
    assert blended.tolist() == pytest.approx([0.3, 0.7])


def test_find_best_blend_weight_prefers_better_model() -> None:
    best_weight, grid = find_best_blend_weight(
        [0.9, 0.1, 0.8, 0.2],
        [0.6, 0.4, 0.6, 0.4],
        [1, 0, 1, 0],
        step=0.5,
    )
    assert best_weight == pytest.approx(1.0)
    assert {"first_weight", "flat_brier"}.issubset(grid.columns)


def test_merge_prediction_frames_joins_on_canonical_game_keys() -> None:
    first = pd.DataFrame(
        {
            "Season": [2024],
            "LowTeamID": [10],
            "HighTeamID": [20],
            "outcome": [1],
            "round_group": ["R1"],
            "pred": [0.7],
        }
    )
    second = pd.DataFrame(
        {
            "Season": [2024],
            "LowTeamID": [10],
            "HighTeamID": [20],
            "outcome": [1],
            "round_group": ["R1"],
            "pred": [0.6],
        }
    )

    merged = merge_prediction_frames(first, second)

    assert merged["pred_first"].tolist() == [0.7]
    assert merged["pred_second"].tolist() == [0.6]
    assert merged["round_group"].tolist() == ["R1"]
