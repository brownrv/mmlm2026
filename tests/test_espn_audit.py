from __future__ import annotations

import pandas as pd
import pytest

from mmlm2026.analysis.espn_audit import (
    summarize_espn_coverage,
    summarize_espn_null_profile,
    summarize_espn_viability,
)


def test_espn_audit_summaries() -> None:
    feature_frames = pd.DataFrame(
        {
            "league": ["M", "M", "W"],
            "feature_group": ["four_factor", "rotation", "four_factor"],
            "Season": [2024, 2024, 2024],
            "TeamID": [1, 1, 10],
            "espn_four_factor_strength": [0.5, pd.NA, 0.7],
            "espn_rotation_stability": [pd.NA, 1.2, pd.NA],
        }
    )
    team_universe = pd.DataFrame(
        {
            "league": ["M", "M", "W", "W"],
            "Season": [2024, 2024, 2024, 2024],
            "TeamID": [1, 2, 10, 20],
        }
    )

    coverage = summarize_espn_coverage(feature_frames, team_universe)
    men_four = coverage.loc[
        (coverage["league"] == "M") & (coverage["feature_group"] == "four_factor")
    ].iloc[0]
    assert men_four["covered_teams"] == 1
    assert men_four["coverage_rate"] == pytest.approx(0.5)

    null_profile = summarize_espn_null_profile(feature_frames)
    women_four = null_profile.loc[
        (null_profile["league"] == "W")
        & (null_profile["feature_group"] == "four_factor")
        & (null_profile["feature_name"] == "espn_four_factor_strength")
    ].iloc[0]
    assert women_four["non_null_rate"] == pytest.approx(1.0)

    viability = summarize_espn_viability(coverage, null_profile)
    men_rotation = viability.loc[
        (viability["league"] == "M") & (viability["feature_group"] == "rotation")
    ].iloc[0]
    assert men_rotation["overall_coverage_rate"] == pytest.approx(0.5)
