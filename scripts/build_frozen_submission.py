from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mmlm2026.evaluation.bracket import save_bracket_artifacts
from mmlm2026.submission.frozen import (
    assign_submission_league,
    default_submission_output_path,
    parse_submission_ids,
)
from mmlm2026.submission.frozen_models import (
    predict_frozen_men,
    predict_frozen_women,
)
from mmlm2026.submission.validation import validate_submission_file
from mmlm2026.submission.writer import write_submission


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the combined frozen-model submission.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026/SampleSubmissionStage2.csv"),
    )
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/submissions"))
    parser.add_argument("--tag", default="frozen")
    parser.add_argument("--save-artifacts", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    sample_submission = pd.read_csv(args.sample_submission)
    parsed = parse_submission_ids(sample_submission)
    season_values = sorted(int(value) for value in parsed["Season"].unique())
    season = args.season if args.season is not None else season_values[0]
    if season_values != [season]:
        raise ValueError(
            "Sample submission must contain exactly one season. "
            f"Found {season_values}, requested {season}."
        )

    men_teams = pd.read_csv(args.data_dir / "MTeams.csv")
    women_teams = pd.read_csv(args.data_dir / "WTeams.csv")
    parsed = assign_submission_league(
        parsed,
        men_team_ids=set(men_teams["TeamID"].astype(int)),
        women_team_ids=set(women_teams["TeamID"].astype(int)),
    )

    men_rows = parsed.loc[
        parsed["league"] == "M",
        ["ID", "Season", "LowTeamID", "HighTeamID"],
    ].copy()
    women_rows = parsed.loc[
        parsed["league"] == "W",
        ["ID", "Season", "LowTeamID", "HighTeamID"],
    ].copy()

    men_predictions, men_bracket = predict_frozen_men(args.data_dir, men_rows, season)
    women_predictions, women_bracket = predict_frozen_women(args.data_dir, women_rows, season)

    combined = (
        pd.concat(
            [men_predictions[["ID", "Pred"]], women_predictions[["ID", "Pred"]]],
            ignore_index=True,
        )
        .sort_values("ID")
        .reset_index(drop=True)
    )

    output_path = default_submission_output_path(
        season=season,
        output_dir=args.output_dir,
        tag=args.tag,
    )
    write_submission(combined, output_path)
    validate_submission_file(output_path, sample_submission_path=args.sample_submission)

    if args.save_artifacts:
        artifacts_dir = args.output_dir / f"{season}_{args.tag}_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        men_predictions.to_parquet(artifacts_dir / "men_predictions.parquet", index=False)
        women_predictions.to_parquet(artifacts_dir / "women_predictions.parquet", index=False)
        save_bracket_artifacts(men_bracket, output_dir=artifacts_dir / "men_bracket")
        save_bracket_artifacts(women_bracket, output_dir=artifacts_dir / "women_bracket")

    print(f"Wrote validated submission to {output_path}")
    print(f"Men rows: {len(men_predictions)} | Women rows: {len(women_predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
