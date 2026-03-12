from __future__ import annotations

import pandas as pd
import pytest

from mmlm2026.submission.frozen import (
    assign_submission_league,
    default_submission_output_path,
    parse_submission_ids,
)


def test_parse_submission_ids_extracts_season_and_teams() -> None:
    frame = pd.DataFrame({"ID": ["2026_1101_1203"]})
    parsed = parse_submission_ids(frame)
    assert parsed.loc[0, "Season"] == 2026
    assert parsed.loc[0, "LowTeamID"] == 1101
    assert parsed.loc[0, "HighTeamID"] == 1203


def test_assign_submission_league_maps_rows_to_single_league() -> None:
    frame = pd.DataFrame({"LowTeamID": [1101, 3101], "HighTeamID": [1203, 3203]})
    assigned = assign_submission_league(
        frame,
        men_team_ids={1101, 1203},
        women_team_ids={3101, 3203},
    )
    assert assigned["league"].tolist() == ["M", "W"]


def test_assign_submission_league_rejects_ambiguous_rows() -> None:
    frame = pd.DataFrame({"LowTeamID": [1101], "HighTeamID": [3101]})
    with pytest.raises(ValueError, match="Unable to assign league"):
        assign_submission_league(
            frame,
            men_team_ids={1101},
            women_team_ids={3101},
        )


def test_default_submission_output_path_uses_season_and_tag() -> None:
    path = default_submission_output_path(season=2026, output_dir="data/submissions", tag="frozen")
    assert path.as_posix().endswith("data/submissions/2026_frozen_submission.csv")
