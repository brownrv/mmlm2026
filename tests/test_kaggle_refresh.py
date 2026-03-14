from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmlm2026.data.kaggle_refresh import load_detailed_results_with_refresh


def test_load_detailed_results_with_refresh_replaces_revised_seasons(
    monkeypatch: object,
) -> None:
    data_dir = Path("data/raw/march-machine-learning-mania-2026")
    base = pd.DataFrame(
        [
            {"Season": 2020, "DayNum": 10, "value": 1},
            {"Season": 2021, "DayNum": 11, "value": 2},
        ]
    )
    revised = pd.DataFrame(
        [
            {"Season": 2021, "DayNum": 12, "value": 99},
            {"Season": 2022, "DayNum": 13, "value": 3},
        ]
    )

    monkeypatch.setattr(Path, "exists", lambda self: self.name == "revised.csv")

    def fake_read_csv(path: Path) -> pd.DataFrame:
        if path.name == "base.csv":
            return base.copy()
        if path.name == "revised.csv":
            return revised.copy()
        raise AssertionError(f"Unexpected path {path}")

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    loaded = load_detailed_results_with_refresh(
        data_dir,
        base_filename="base.csv",
        revised_filename="revised.csv",
    )

    assert loaded["Season"].tolist() == [2020, 2021, 2022]
    assert loaded["value"].tolist() == [1, 99, 3]


def test_load_detailed_results_with_refresh_falls_back_to_base(monkeypatch: object) -> None:
    data_dir = Path("data/raw/march-machine-learning-mania-2026")
    base = pd.DataFrame(
        [
            {"Season": 2020, "DayNum": 10, "value": 1},
            {"Season": 2021, "DayNum": 11, "value": 2},
        ]
    )

    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr(pd, "read_csv", lambda path: base.copy())

    loaded = load_detailed_results_with_refresh(
        data_dir,
        base_filename="base.csv",
        revised_filename="missing.csv",
    )

    assert loaded.equals(base)
