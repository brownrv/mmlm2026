from __future__ import annotations

import pandas as pd


def season_holdout_split(
    frame: pd.DataFrame, season_col: str, holdout_season: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame into train/valid by held-out season."""
    train = frame.loc[frame[season_col] != holdout_season].copy()
    valid = frame.loc[frame[season_col] == holdout_season].copy()
    return train, valid
