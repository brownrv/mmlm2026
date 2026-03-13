from __future__ import annotations

import pandas as pd

from mmlm2026.submission.frozen_models import build_seeded_submission_rows


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
