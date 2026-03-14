from __future__ import annotations

import pandas as pd

from mmlm2026.analysis.revised_detailed_audit import (
    build_refresh_dependency_matrix,
    summarize_revised_vs_original,
)


def _sample_detailed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Season": 2021,
                "DayNum": 10,
                "WTeamID": 1101,
                "WScore": 70,
                "LTeamID": 1201,
                "LScore": 60,
                "WLoc": "H",
                "NumOT": 0,
                "WFGM": 24,
                "WFGA": 50,
                "WFGM3": 6,
                "WFGA3": 20,
                "WFTM": 16,
                "WFTA": 20,
                "WOR": 10,
                "WDR": 20,
                "WAst": 12,
                "WTO": 9,
                "WStl": 7,
                "WBlk": 4,
                "WPF": 14,
                "LFGM": 21,
                "LFGA": 49,
                "LFGM3": 4,
                "LFGA3": 18,
                "LFTM": 14,
                "LFTA": 18,
                "LOR": 8,
                "LDR": 17,
                "LAst": 10,
                "LTO": 11,
                "LStl": 5,
                "LBlk": 3,
                "LPF": 16,
            },
            {
                "Season": 2022,
                "DayNum": 11,
                "WTeamID": 1102,
                "WScore": 75,
                "LTeamID": 1202,
                "LScore": 65,
                "WLoc": "N",
                "NumOT": 0,
                "WFGM": 26,
                "WFGA": 55,
                "WFGM3": 7,
                "WFGA3": 22,
                "WFTM": 16,
                "WFTA": 19,
                "WOR": 11,
                "WDR": 22,
                "WAst": 13,
                "WTO": 10,
                "WStl": 8,
                "WBlk": 5,
                "WPF": 13,
                "LFGM": 22,
                "LFGA": 50,
                "LFGM3": 5,
                "LFGA3": 19,
                "LFTM": 16,
                "LFTA": 20,
                "LOR": 9,
                "LDR": 18,
                "LAst": 11,
                "LTO": 12,
                "LStl": 4,
                "LBlk": 2,
                "LPF": 15,
            },
        ]
    )


def test_summarize_revised_vs_original_detects_changed_and_added_games() -> None:
    original = _sample_detailed_frame()
    revised = original.copy()
    revised.loc[0, "WBlk"] = 6
    revised = pd.concat(
        [
            revised,
            pd.DataFrame(
                [
                    {
                        "Season": 2023,
                        "DayNum": 12,
                        "WTeamID": 1103,
                        "WScore": 80,
                        "LTeamID": 1203,
                        "LScore": 70,
                        "WLoc": "A",
                        "NumOT": 0,
                        "WFGM": 27,
                        "WFGA": 58,
                        "WFGM3": 8,
                        "WFGA3": 23,
                        "WFTM": 18,
                        "WFTA": 22,
                        "WOR": 12,
                        "WDR": 21,
                        "WAst": 14,
                        "WTO": 8,
                        "WStl": 6,
                        "WBlk": 3,
                        "WPF": 12,
                        "LFGM": 24,
                        "LFGA": 53,
                        "LFGM3": 6,
                        "LFGA3": 20,
                        "LFTM": 16,
                        "LFTA": 18,
                        "LOR": 10,
                        "LDR": 19,
                        "LAst": 12,
                        "LTO": 13,
                        "LStl": 5,
                        "LBlk": 2,
                        "LPF": 14,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    season_summary, stat_summary = summarize_revised_vs_original(
        original=original,
        revised=revised,
        league="M",
        scope="regular",
    )

    season_2021 = season_summary.loc[season_summary["Season"] == 2021].iloc[0]
    assert season_2021["changed_games"] == 1
    season_2023 = season_summary.loc[season_summary["Season"] == 2023].iloc[0]
    assert season_2023["added_games"] == 1

    wblk_2021 = stat_summary.loc[
        (stat_summary["Season"] == 2021) & (stat_summary["stat"] == "WBlk")
    ].iloc[0]
    assert wblk_2021["changed_games"] == 1
    assert wblk_2021["max_abs_diff"] == 2.0


def test_build_refresh_dependency_matrix_marks_mh7_feat_02_as_must_retest() -> None:
    matrix = build_refresh_dependency_matrix()
    row = matrix.loc[matrix["item_id"] == "MH7-FEAT-02"].iloc[0]
    assert bool(row["depends_on_revised_detailed"]) is True
    assert row["status"] == "Must re-test"
