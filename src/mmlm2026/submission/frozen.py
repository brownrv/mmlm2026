from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_submission_ids(
    submission: pd.DataFrame,
    *,
    id_col: str = "ID",
) -> pd.DataFrame:
    """Parse Kaggle submission IDs into season and team columns."""
    if id_col not in submission.columns:
        raise ValueError(f"Submission frame missing ID column: {id_col}")

    parts = submission[id_col].astype(str).str.split("_", expand=True)
    if parts.shape[1] != 3:
        raise ValueError("Submission IDs must have the form Season_LowTeamID_HighTeamID.")

    parsed = submission.copy()
    parsed["Season"] = parts[0].astype(int)
    parsed["LowTeamID"] = parts[1].astype(int)
    parsed["HighTeamID"] = parts[2].astype(int)
    return parsed


def assign_submission_league(
    frame: pd.DataFrame,
    *,
    men_team_ids: set[int],
    women_team_ids: set[int],
) -> pd.DataFrame:
    """Assign league to parsed submission rows based on team ID membership."""
    required = {"LowTeamID", "HighTeamID"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Submission frame missing required columns: {sorted(missing)}")

    leagues: list[str] = []
    for low_team, high_team in zip(frame["LowTeamID"], frame["HighTeamID"], strict=False):
        low_team = int(low_team)
        high_team = int(high_team)
        is_men = low_team in men_team_ids and high_team in men_team_ids
        is_women = low_team in women_team_ids and high_team in women_team_ids
        if is_men == is_women:
            raise ValueError(
                "Unable to assign league for submission row "
                f"{low_team}_{high_team}; expected both teams in exactly one league."
            )
        leagues.append("M" if is_men else "W")

    assigned = frame.copy()
    assigned["league"] = leagues
    return assigned


def default_submission_output_path(
    *,
    season: int,
    output_dir: str | Path,
    tag: str = "frozen",
) -> Path:
    """Return the default output path for a final combined submission."""
    return Path(output_dir) / f"{season}_{tag}_submission.csv"
