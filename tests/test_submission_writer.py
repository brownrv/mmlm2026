from __future__ import annotations

import pandas as pd
import pytest

from mmlm2026.submission.writer import write_submission


def test_write_submission_clips_and_writes(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "ID": ["2025_1101_1203", "2025_1102_1304"],
            "Pred": [0.0, 1.0],
        }
    )
    out_path = tmp_path / "submissions" / "submission.csv"
    write_submission(frame, out_path)
    written = pd.read_csv(out_path)
    assert list(written.columns) == ["ID", "Pred"]
    assert written["Pred"].iloc[0] == 0.025
    assert written["Pred"].iloc[1] == 0.975


def test_write_submission_raises_for_missing_columns(tmp_path) -> None:
    frame = pd.DataFrame({"ID": ["2025_1101_1203"]})
    out_path = tmp_path / "submission.csv"
    with pytest.raises(ValueError, match="Missing required submission columns"):
        write_submission(frame, out_path)
