from __future__ import annotations

import argparse
from pathlib import Path

from mmlm2026.submission.validation import validate_submission_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Kaggle submission CSV.")
    parser.add_argument("submission_path", type=Path, help="Path to the submission CSV.")
    parser.add_argument(
        "--sample",
        dest="sample_submission_path",
        type=Path,
        default=None,
        help="Optional sample submission CSV to validate ID coverage and row count.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    result = validate_submission_file(
        args.submission_path,
        sample_submission_path=args.sample_submission_path,
    )

    if result.sample_row_count is None:
        print(
            f"Submission valid: {result.row_count} rows, schema/ID/range checks passed.",
        )
    else:
        print(
            "Submission valid: "
            f"{result.row_count} rows, sample row count {result.sample_row_count}, "
            "schema/ID/range/sample checks passed.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
