from __future__ import annotations

import pandas as pd

from mmlm2026.evaluation.women_alpha import select_alpha_from_prior_seasons


def test_fit_alpha_from_prior_seasons_recovers_elo_weight_when_elo_is_better() -> None:
    frame = pd.DataFrame(
        {
            "Season": [2019, 2019, 2020, 2020, 2021, 2021],
            "women_hca_adj_qg_diff": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "seed_diff": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "close_win_pct_diff": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "elo_diff": [300.0, -300.0, 300.0, -300.0, 300.0, -300.0],
            "outcome": [1, 0, 1, 0, 1, 0],
        }
    )

    alpha = select_alpha_from_prior_seasons(
        frame=frame,
        feature_cols=["women_hca_adj_qg_diff", "seed_diff", "close_win_pct_diff"],
        target_season=2021,
        elo_scale=400.0,
        alpha_fallback=0.10,
        alpha_step=0.05,
        alpha_calibration_seasons=8,
        train_min_games=1,
    )

    assert alpha == 1.0
