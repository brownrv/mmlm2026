from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV into a DataFrame with a consistent path interface."""
    return pd.read_csv(str(Path(path)), **kwargs)
