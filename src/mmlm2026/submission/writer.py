from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_submission(
    submission: pd.DataFrame,
    output_path: str | Path,
    id_col: str = "ID",
    pred_col: str = "Pred",
    clip_min: float = 0.025,
    clip_max: float = 0.975,
) -> Path:
    """Write Kaggle submission file with required schema and clipping."""
    required = {id_col, pred_col}
    missing = required.difference(submission.columns)
    if missing:
        raise ValueError(f"Missing required submission columns: {sorted(missing)}")

    out = submission[[id_col, pred_col]].copy()
    out[pred_col] = out[pred_col].clip(lower=clip_min, upper=clip_max)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path
