from __future__ import annotations

import pandas as pd
import pytest

from mmlm2026.submission.validation import (
    SubmissionValidationResult,
    validate_submission_file,
    validate_submission_frame,
)


def test_validate_submission_frame_accepts_valid_submission() -> None:
    frame = pd.DataFrame(
        {
            "ID": ["2025_1101_1203", "2025_1102_1304"],
            "Pred": [0.25, 0.75],
        }
    )

    result = validate_submission_frame(frame)

    assert result == SubmissionValidationResult(row_count=2, sample_row_count=None)


def test_validate_submission_frame_rejects_bad_id_order() -> None:
    frame = pd.DataFrame({"ID": ["2025_1203_1101"], "Pred": [0.4]})

    with pytest.raises(ValueError, match="LowTeamID is not lower"):
        validate_submission_frame(frame)


def test_validate_submission_frame_rejects_out_of_range_predictions() -> None:
    frame = pd.DataFrame({"ID": ["2025_1101_1203"], "Pred": [0.99]})

    with pytest.raises(ValueError, match=r"within \[0.025, 0.975\]"):
        validate_submission_frame(frame)


def test_validate_submission_frame_rejects_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "ID": ["2025_1101_1203", "2025_1101_1203"],
            "Pred": [0.4, 0.5],
        }
    )

    with pytest.raises(ValueError, match="Duplicate submission IDs"):
        validate_submission_frame(frame)


def test_validate_submission_file_checks_sample_coverage(tmp_path) -> None:
    submission_path = tmp_path / "submission.csv"
    sample_path = tmp_path / "sample.csv"

    pd.DataFrame(
        {
            "ID": ["2025_1101_1203", "2025_1102_1304"],
            "Pred": [0.25, 0.75],
        }
    ).to_csv(submission_path, index=False)
    pd.DataFrame({"ID": ["2025_1101_1203", "2025_1102_1304"]}).to_csv(sample_path, index=False)

    result = validate_submission_file(submission_path, sample_submission_path=sample_path)

    assert result == SubmissionValidationResult(row_count=2, sample_row_count=2)


def test_validate_submission_file_rejects_missing_sample_ids(tmp_path) -> None:
    submission_path = tmp_path / "submission.csv"
    sample_path = tmp_path / "sample.csv"

    pd.DataFrame({"ID": ["2025_1101_1203"], "Pred": [0.25]}).to_csv(submission_path, index=False)
    pd.DataFrame({"ID": ["2025_1101_1203", "2025_1102_1304"]}).to_csv(sample_path, index=False)

    with pytest.raises(ValueError, match="missing IDs from the sample submission"):
        validate_submission_file(submission_path, sample_submission_path=sample_path)
