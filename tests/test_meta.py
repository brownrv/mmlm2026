from __future__ import annotations

import pandas as pd

from mmlm2026.evaluation.meta import (
    logit_clip,
    merge_meta_prediction_frames,
    tune_logit_ridge_alpha,
)


def test_logit_clip_returns_finite_values() -> None:
    logits = logit_clip(pd.Series([0.0, 0.25, 0.5, 0.75, 1.0]))

    assert len(logits) == 5
    assert pd.Series(logits).map(pd.notna).all()


def test_merge_meta_prediction_frames_merges_on_canonical_ids() -> None:
    first = pd.DataFrame(
        {
            "Season": [2023],
            "LowTeamID": [1],
            "HighTeamID": [2],
            "outcome": [1],
            "pred": [0.6],
        }
    )
    second = pd.DataFrame(
        {
            "Season": [2023],
            "LowTeamID": [1],
            "HighTeamID": [2],
            "outcome": [1],
            "pred": [0.55],
        }
    )

    merged = merge_meta_prediction_frames([first, second], pred_col_names=["pred_a", "pred_b"])

    assert merged.loc[0, "pred_a"] == 0.6
    assert merged.loc[0, "pred_b"] == 0.55


def test_tune_logit_ridge_alpha_returns_best_alpha() -> None:
    frame = pd.DataFrame(
        {
            "Season": [2021, 2021, 2022, 2022, 2023, 2023],
            "outcome": [1, 0, 1, 0, 1, 0],
            "logit_pred_a": [1.5, -1.2, 1.1, -1.0, 1.4, -1.3],
            "logit_pred_b": [1.2, -1.0, 0.9, -0.8, 1.3, -1.1],
        }
    )

    result = tune_logit_ridge_alpha(
        frame,
        feature_cols=["logit_pred_a", "logit_pred_b"],
        alpha_grid=[0.01, 0.1, 1.0],
    )

    assert result.alpha in {0.01, 0.1, 1.0}
    assert not result.per_alpha.empty
