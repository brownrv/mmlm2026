from __future__ import annotations

import pandas as pd

from mmlm2026.evaluation.splits import season_holdout_split


def test_season_holdout_split() -> None:
    frame = pd.DataFrame(
        {
            "Season": [2023, 2024, 2024, 2025],
            "value": [1, 2, 3, 4],
        }
    )
    train, valid = season_holdout_split(frame, season_col="Season", holdout_season=2024)
    assert set(train["Season"]) == {2023, 2025}
    assert set(valid["Season"]) == {2024}
