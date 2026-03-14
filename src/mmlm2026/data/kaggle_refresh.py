from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_detailed_results_with_refresh(
    data_dir: Path,
    *,
    base_filename: str,
    revised_filename: str,
) -> pd.DataFrame:
    """Load a detailed-results file, replacing any revised seasons when available."""
    base_path = data_dir / base_filename
    revised_path = data_dir / revised_filename

    base = pd.read_csv(base_path)
    if not revised_path.exists():
        return base

    revised = pd.read_csv(revised_path)
    revised_seasons = set(revised["Season"].astype(int).unique().tolist())
    base = base.loc[~base["Season"].astype(int).isin(revised_seasons)].copy()
    combined = pd.concat([base, revised], ignore_index=True)
    return combined.sort_values(["Season", "DayNum"]).reset_index(drop=True)
