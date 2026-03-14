from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmlm2026.features.espn import (
    load_espn_men_four_factor_strength_features,
    load_espn_men_rotation_stability_features,
    load_espn_women_four_factor_strength_features,
)


def load_espn_feature_frames(
    *,
    data_dir: Path,
    espn_root: Path,
    team_spellings_path: Path,
    season_floor: int = 2004,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ESPN-derived features and season-level team universes for audit."""
    men_regular = pd.read_csv(data_dir / "MRegularSeasonDetailedResults.csv")
    women_regular = pd.read_csv(data_dir / "WRegularSeasonDetailedResults.csv")
    men_regular = men_regular.loc[men_regular["Season"] >= season_floor].copy()
    women_regular = women_regular.loc[women_regular["Season"] >= season_floor].copy()

    men_seasons = sorted(men_regular["Season"].astype(int).unique().tolist())
    women_seasons = sorted(women_regular["Season"].astype(int).unique().tolist())

    men_four_factor = load_espn_men_four_factor_strength_features(
        espn_root=espn_root / "mens-college-basketball",
        seasons=men_seasons,
        regular_season_results=men_regular,
        team_spellings_path=team_spellings_path,
    ).assign(league="M", feature_group="four_factor")
    women_four_factor = load_espn_women_four_factor_strength_features(
        espn_root=espn_root / "womens-college-basketball",
        seasons=women_seasons,
        regular_season_results=women_regular,
        team_spellings_path=team_spellings_path,
    ).assign(league="W", feature_group="four_factor")
    men_rotation = load_espn_men_rotation_stability_features(
        espn_root=espn_root / "mens-college-basketball",
        seasons=men_seasons,
        regular_season_results=men_regular,
        team_spellings_path=team_spellings_path,
    ).assign(league="M", feature_group="rotation")

    feature_frames = pd.concat(
        [men_four_factor, women_four_factor, men_rotation],
        ignore_index=True,
        sort=False,
    )

    team_universe = pd.concat(
        [
            _build_team_universe(men_regular, league="M"),
            _build_team_universe(women_regular, league="W"),
        ],
        ignore_index=True,
    )
    return feature_frames, team_universe


def _build_team_universe(regular_season_results: pd.DataFrame, *, league: str) -> pd.DataFrame:
    winners = regular_season_results[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"})
    losers = regular_season_results[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"})
    universe = pd.concat([winners, losers], ignore_index=True).drop_duplicates()
    universe["league"] = league
    universe["Season"] = universe["Season"].astype(int)
    universe["TeamID"] = universe["TeamID"].astype(int)
    return universe.sort_values(["league", "Season", "TeamID"]).reset_index(drop=True)


def summarize_espn_coverage(
    feature_frames: pd.DataFrame,
    team_universe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize season-level ESPN feature coverage against the regular-season team universe."""
    universe_counts = (
        team_universe.groupby(["league", "Season"], as_index=False)
        .agg(universe_teams=("TeamID", "nunique"))
        .sort_values(["league", "Season"])
        .reset_index(drop=True)
    )

    grouped = []
    for (league, feature_group), feature_frame in feature_frames.groupby(
        ["league", "feature_group"], dropna=False
    ):
        seasons = universe_counts.loc[universe_counts["league"] == league, "Season"].tolist()
        covered = (
            feature_frame.groupby("Season", as_index=False)
            .agg(covered_teams=("TeamID", "nunique"))
            .assign(league=league, feature_group=feature_group)
        )
        base = pd.DataFrame(
            {
                "league": league,
                "feature_group": feature_group,
                "Season": seasons,
            }
        )
        grouped.append(base.merge(covered, on=["league", "feature_group", "Season"], how="left"))

    coverage = pd.concat(grouped, ignore_index=True).merge(
        universe_counts,
        on=["league", "Season"],
        how="left",
        validate="many_to_one",
    )
    coverage["covered_teams"] = coverage["covered_teams"].fillna(0).astype(int)
    coverage["missing_teams"] = coverage["universe_teams"] - coverage["covered_teams"]
    coverage["coverage_rate"] = coverage["covered_teams"] / coverage["universe_teams"]
    coverage["missing_rate"] = 1.0 - coverage["coverage_rate"]
    return coverage.sort_values(["league", "feature_group", "Season"]).reset_index(drop=True)


def summarize_espn_null_profile(feature_frames: pd.DataFrame) -> pd.DataFrame:
    """Summarize season-level per-feature non-null rates for ESPN-derived features."""
    id_cols = {"league", "feature_group", "Season", "TeamID"}
    rows: list[dict[str, float | int | str]] = []
    for (league, feature_group, season), season_frame in feature_frames.groupby(
        ["league", "feature_group", "Season"], dropna=False
    ):
        feature_cols = [
            col
            for col in season_frame.columns
            if col not in id_cols and season_frame[col].notna().any()
        ]
        for feature_name in feature_cols:
            non_null_rate = float(season_frame[feature_name].notna().mean())
            rows.append(
                {
                    "league": str(league),
                    "feature_group": str(feature_group),
                    "Season": int(season),
                    "feature_name": feature_name,
                    "rows": int(len(season_frame)),
                    "non_null_rate": non_null_rate,
                    "null_rate": 1.0 - non_null_rate,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["league", "feature_group", "Season", "feature_name"])
        .reset_index(drop=True)
    )


def summarize_espn_viability(coverage: pd.DataFrame, null_profile: pd.DataFrame) -> pd.DataFrame:
    """Aggregate high-level viability signals by league and feature family."""
    null_summary = (
        null_profile.groupby(["league", "feature_group"], as_index=False)
        .agg(
            mean_non_null_rate=("non_null_rate", "mean"),
            min_non_null_rate=("non_null_rate", "min"),
        )
        .sort_values(["league", "feature_group"])
        .reset_index(drop=True)
    )
    viability = (
        coverage.groupby(["league", "feature_group"], as_index=False)
        .agg(
            seasons=("Season", "nunique"),
            total_universe_teams=("universe_teams", "sum"),
            total_covered_teams=("covered_teams", "sum"),
            mean_coverage_rate=("coverage_rate", "mean"),
            median_coverage_rate=("coverage_rate", "median"),
            min_coverage_rate=("coverage_rate", "min"),
            max_coverage_rate=("coverage_rate", "max"),
        )
        .sort_values(["league", "feature_group"])
        .reset_index(drop=True)
    )
    viability["overall_coverage_rate"] = (
        viability["total_covered_teams"] / viability["total_universe_teams"]
    )
    return viability.merge(
        null_summary,
        on=["league", "feature_group"],
        how="left",
        validate="one_to_one",
    )
