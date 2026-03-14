from __future__ import annotations

import pandas as pd

from mmlm2026.submission.frozen_models import (
    _select_nonempty_feature_cols,
    build_seeded_submission_rows,
)


def test_build_seeded_submission_rows_returns_all_unique_pairs() -> None:
    seeds = pd.DataFrame(
        {
            "Season": [2025, 2025, 2025],
            "Seed": ["W01", "W16", "X08"],
            "TeamID": [1111, 1222, 1333],
        }
    )

    rows = build_seeded_submission_rows(seeds, season=2025)

    assert rows["ID"].tolist() == [
        "2025_1111_1222",
        "2025_1111_1333",
        "2025_1222_1333",
    ]
    assert rows[["Season", "LowTeamID", "HighTeamID"]].values.tolist() == [
        [2025, 1111, 1222],
        [2025, 1111, 1333],
        [2025, 1222, 1333],
    ]


def test_select_nonempty_feature_cols_drops_all_missing_columns() -> None:
    frame = pd.DataFrame(
        {
            "seed_diff": [1.0, 2.0],
            "elo_diff": [10.0, 12.0],
            "espn_four_factor_strength_diff": [float("nan"), float("nan")],
            "conf_pct_rank_diff": [0.2, 0.4],
        }
    )

    active = _select_nonempty_feature_cols(
        frame,
        [
            "seed_diff",
            "elo_diff",
            "espn_four_factor_strength_diff",
            "conf_pct_rank_diff",
        ],
    )

    assert active == ["seed_diff", "elo_diff", "conf_pct_rank_diff"]
