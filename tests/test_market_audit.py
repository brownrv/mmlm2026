from __future__ import annotations

import pandas as pd
import pytest

from mmlm2026.analysis.market_audit import (
    compare_market_to_local_tourney,
    summarize_market_coverage,
    summarize_market_viability,
)


def test_market_audit_summaries_and_compares() -> None:
    games = pd.DataFrame(
        {
            "league": ["M", "M", "W"],
            "stage": ["tourney", "regular", "tourney"],
            "Season": [2024, 2024, 2024],
            "ID": ["2024_1_2", "2024_1_3", "2024_10_20"],
            "has_market": [True, False, True],
            "prob_sum_error": [0.0, 0.0, 0.0],
            "prob_out_of_range": [False, False, False],
            "market_pred": [0.8, pd.NA, 0.4],
            "outcome": [1.0, 0.0, 0.0],
        }
    )
    local = pd.DataFrame(
        {
            "league": ["M", "W"],
            "Season": [2024, 2024],
            "ID": ["2024_1_2", "2024_10_20"],
            "Pred": [0.7, 0.6],
            "outcome": [1.0, 0.0],
            "brier_component": [0.09, 0.36],
        }
    )

    coverage = summarize_market_coverage(games)
    assert coverage["games"].sum() == 3
    assert coverage["games_with_market"].sum() == 2

    viability = summarize_market_viability(coverage)
    assert set(viability["league"]) == {"M", "W"}

    comparison = compare_market_to_local_tourney(games, local)
    men_2024 = comparison.loc[(comparison["league"] == "M") & (comparison["Season"] == 2024)].iloc[
        0
    ]
    assert men_2024["compared_games"] == 1
    assert men_2024["market_brier"] == pytest.approx(0.04)
    assert men_2024["local_brier"] == pytest.approx(0.09)
