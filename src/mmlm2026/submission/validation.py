from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_ID_PATTERN = re.compile(r"^(?P<season>\d{4})_(?P<low>\d+)_(?P<high>\d+)$")


@dataclass(frozen=True)
class SubmissionValidationResult:
    """Structured result for submission validation."""

    row_count: int
    sample_row_count: int | None = None


def validate_submission_frame(
    submission: pd.DataFrame,
    *,
    sample_submission: pd.DataFrame | None = None,
    id_col: str = "ID",
    pred_col: str = "Pred",
    clip_min: float = 0.025,
    clip_max: float = 0.975,
) -> SubmissionValidationResult:
    """Validate a submission frame against project submission rules."""
    required = {id_col, pred_col}
    missing = required.difference(submission.columns)
    if missing:
        raise ValueError(f"Missing required submission columns: {sorted(missing)}")

    frame = submission[[id_col, pred_col]].copy()

    if frame.empty:
        raise ValueError("Submission is empty.")

    duplicate_mask = frame[id_col].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = sorted(frame.loc[duplicate_mask, id_col].astype(str).unique().tolist())
        preview = duplicates[:5]
        raise ValueError(f"Duplicate submission IDs found: {preview}")

    if frame[pred_col].isna().any():
        raise ValueError("Submission contains missing prediction values.")

    pred_numeric = pd.to_numeric(frame[pred_col], errors="coerce")
    if pred_numeric.isna().any():
        raise ValueError("Submission contains non-numeric prediction values.")

    out_of_range = ~pred_numeric.between(clip_min, clip_max, inclusive="both")
    if out_of_range.any():
        bad_values = pred_numeric.loc[out_of_range].tolist()[:5]
        raise ValueError(
            f"Submission predictions must be within [{clip_min}, {clip_max}]. "
            f"Found out-of-range values: {bad_values}"
        )

    invalid_ids: list[str] = []
    misordered_ids: list[str] = []
    for raw_id in frame[id_col].astype(str):
        match = _ID_PATTERN.fullmatch(raw_id)
        if match is None:
            invalid_ids.append(raw_id)
            continue
        low_team = int(match.group("low"))
        high_team = int(match.group("high"))
        if low_team >= high_team:
            misordered_ids.append(raw_id)

    if invalid_ids:
        raise ValueError(
            "Submission contains IDs with invalid format. "
            f"Expected YYYY_LowTeamID_HighTeamID. Examples: {invalid_ids[:5]}"
        )

    if misordered_ids:
        raise ValueError(
            "Submission contains IDs where LowTeamID is not lower than HighTeamID. "
            f"Examples: {misordered_ids[:5]}"
        )

    sample_row_count: int | None = None
    if sample_submission is not None:
        sample_missing = {id_col}.difference(sample_submission.columns)
        if sample_missing:
            raise ValueError(
                f"Sample submission is missing required ID column: {sorted(sample_missing)}"
            )

        sample_ids = sample_submission[id_col].astype(str)
        submission_ids = frame[id_col].astype(str)
        sample_row_count = len(sample_ids)

        missing_ids = sorted(set(sample_ids).difference(submission_ids))
        extra_ids = sorted(set(submission_ids).difference(sample_ids))
        if missing_ids:
            raise ValueError(
                f"Submission is missing IDs from the sample submission. Examples: {missing_ids[:5]}"
            )
        if extra_ids:
            raise ValueError(
                "Submission contains IDs not present in the sample submission. "
                f"Examples: {extra_ids[:5]}"
            )
        if len(submission_ids) != len(sample_ids):
            raise ValueError(
                "Submission row count does not match sample submission row count. "
                f"Expected {len(sample_ids)}, found {len(submission_ids)}."
            )

    return SubmissionValidationResult(
        row_count=len(frame),
        sample_row_count=sample_row_count,
    )


def validate_submission_file(
    submission_path: str | Path,
    *,
    sample_submission_path: str | Path | None = None,
    id_col: str = "ID",
    pred_col: str = "Pred",
    clip_min: float = 0.025,
    clip_max: float = 0.975,
) -> SubmissionValidationResult:
    """Load and validate a submission CSV file."""
    submission = pd.read_csv(Path(submission_path))
    sample_submission = None
    if sample_submission_path is not None:
        sample_submission = pd.read_csv(Path(sample_submission_path))
    return validate_submission_frame(
        submission,
        sample_submission=sample_submission,
        id_col=id_col,
        pred_col=pred_col,
        clip_min=clip_min,
        clip_max=clip_max,
    )
