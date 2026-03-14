from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmlm2026.features.espn import (
    _build_unique_espn_mapping,
    _match_espn_games_to_kaggle,
    _normalize_espn_id,
    _pair_espn_games,
)

SEASONS_2021_2026 = list(range(2021, 2027))
DETAILED_BOX_COLS = [
    "FGM",
    "FGA",
    "FGM3",
    "FGA3",
    "FTM",
    "FTA",
    "OR",
    "DR",
    "Ast",
    "TO",
    "Stl",
    "Blk",
    "PF",
]
RAW_DIFF_PRIORITY_STATS = ["WBlk", "LBlk", "WStl", "LStl", "WTO", "LTO", "WOR", "LOR", "WDR", "LDR"]
ESPN_COMPARISON_STATS = [
    "WFGA",
    "LFGA",
    "WAst",
    "LAst",
    "WTO",
    "LTO",
    "WStl",
    "LStl",
    "WBlk",
    "LBlk",
    "WPF",
    "LPF",
    "WOR",
    "LOR",
    "WDR",
    "LDR",
]


def load_revised_detailed_file_inventory(data_dir: Path) -> pd.DataFrame:
    """Build a manifest-style inventory for the revised detailed files."""
    files = [
        "MRegularSeasonDetailedResults_2021_2026.csv",
        "WRegularSeasonDetailedResults_2021_2026.csv",
        "MNCAATourneyDetailedResults_2021_2026.csv",
        "WNCAATourneyDetailedResults_2021_2026.csv",
    ]
    rows: list[dict[str, object]] = []
    for file_name in files:
        path = data_dir / file_name
        frame = pd.read_csv(path)
        rows.append(
            {
                "file_name": file_name,
                "rows": int(len(frame)),
                "season_min": int(frame["Season"].min()),
                "season_max": int(frame["Season"].max()),
                "daynum_min": int(frame["DayNum"].min()),
                "daynum_max": int(frame["DayNum"].max()),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return pd.DataFrame(rows).sort_values("file_name").reset_index(drop=True)


def summarize_revised_vs_original(
    *,
    original: pd.DataFrame,
    revised: pd.DataFrame,
    league: str,
    scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize game- and stat-level differences between original and revised detailed files."""
    original = original.loc[original["Season"].isin(SEASONS_2021_2026)].copy()
    revised = revised.loc[revised["Season"].isin(SEASONS_2021_2026)].copy()

    original_keyed = _add_game_key(original)
    revised_keyed = _add_game_key(revised)
    key_cols = ["Season", "DayNum", "low_team", "high_team"]

    season_rows: list[dict[str, object]] = []
    stat_rows: list[dict[str, object]] = []

    for season in SEASONS_2021_2026:
        original_season = original_keyed.loc[original_keyed["Season"] == season].copy()
        revised_season = revised_keyed.loc[revised_keyed["Season"] == season].copy()
        original_keys = set(_frame_keys(original_season[key_cols]))
        revised_keys = set(_frame_keys(revised_season[key_cols]))
        shared_keys = original_keys & revised_keys
        added_keys = revised_keys - original_keys
        dropped_keys = original_keys - revised_keys

        merged = original_season.merge(
            revised_season,
            on=key_cols,
            suffixes=("_orig", "_rev"),
            how="inner",
            validate="one_to_one",
        )
        row_changed = pd.Series(False, index=merged.index, dtype=bool)
        for col in _tracked_compare_cols(original):
            row_changed = row_changed | (merged[f"{col}_orig"] != merged[f"{col}_rev"])

        season_rows.append(
            {
                "league": league,
                "scope": scope,
                "Season": season,
                "original_games": int(len(original_season)),
                "revised_games": int(len(revised_season)),
                "shared_games": int(len(shared_keys)),
                "added_games": int(len(added_keys)),
                "dropped_games": int(len(dropped_keys)),
                "changed_games": int(row_changed.sum()),
                "unchanged_shared_games": int((~row_changed).sum()),
            }
        )

        for col in _tracked_compare_cols(original):
            if pd.api.types.is_numeric_dtype(original[col]):
                diff = merged[f"{col}_rev"] - merged[f"{col}_orig"]
                mean_abs_diff = float(diff.abs().mean()) if len(diff) else 0.0
                max_abs_diff = float(diff.abs().max()) if len(diff) else 0.0
                net_diff = float(diff.sum()) if len(diff) else 0.0
            else:
                diff = merged[f"{col}_rev"] != merged[f"{col}_orig"]
                mean_abs_diff = float(diff.mean()) if len(diff) else 0.0
                max_abs_diff = 1.0 if bool(diff.any()) else 0.0
                net_diff = float(diff.sum()) if len(diff) else 0.0
            stat_rows.append(
                {
                    "league": league,
                    "scope": scope,
                    "Season": season,
                    "stat": col,
                    "shared_games": int(len(merged)),
                    "changed_games": int((diff != 0).sum()),
                    "mean_abs_diff": mean_abs_diff,
                    "max_abs_diff": max_abs_diff,
                    "net_diff": net_diff,
                    "priority_stat": bool(col in RAW_DIFF_PRIORITY_STATS),
                }
            )

    season_summary = pd.DataFrame(season_rows).sort_values(["league", "scope", "Season"])
    stat_summary = pd.DataFrame(stat_rows).sort_values(
        ["league", "scope", "priority_stat", "Season", "stat"],
        ascending=[True, True, False, True, True],
    )
    return season_summary.reset_index(drop=True), stat_summary.reset_index(drop=True)


def build_refresh_dependency_matrix() -> pd.DataFrame:
    """Enumerate the current refresh-gate dependency status for important local branches."""
    rows = [
        {
            "item_id": "MEN-FREEZE",
            "item_type": "frozen_leader",
            "league": "M",
            "depends_on_revised_detailed": False,
            "status": "Probably safe",
            "reason": (
                "Generalization-tuned men reference margin path is compact/Elo based "
                "in its frozen form."
            ),
        },
        {
            "item_id": "WOMEN-FREEZE",
            "item_type": "frozen_leader",
            "league": "W",
            "depends_on_revised_detailed": False,
            "status": "Probably safe",
            "reason": (
                "Current women COOPER win-bonus + early-k path uses routed women + "
                "ESPN four-factor + conference-rank features, not Kaggle detailed "
                "boxscore aggregates."
            ),
        },
        {
            "item_id": "LATE-RATE-01",
            "item_type": "challenger_branch",
            "league": "M",
            "depends_on_revised_detailed": True,
            "status": "Must re-test",
            "reason": "Ridge-strength slice used detailed regular-season results directly.",
        },
        {
            "item_id": "LATE-RATE-02",
            "item_type": "challenger_branch",
            "league": "W",
            "depends_on_revised_detailed": False,
            "status": "Possibly affected",
            "reason": (
                "Primary winning slice was ESPN four-factor based, but any later "
                "combined branches using detailed-derived extras should be checked."
            ),
        },
        {
            "item_id": "MH7-FEAT-01",
            "item_type": "feature_challenger",
            "league": "M+W",
            "depends_on_revised_detailed": False,
            "status": "Probably safe",
            "reason": (
                "GLM quality coefficients are built from compact regular-season "
                "results in the current implementation, not detailed box scores."
            ),
        },
        {
            "item_id": "MH7-FEAT-02",
            "item_type": "feature_challenger",
            "league": "M+W",
            "depends_on_revised_detailed": True,
            "status": "Must re-test",
            "reason": "Opponent raw box-score averages use detailed results directly.",
        },
        {
            "item_id": "COOPER-FEAT-01",
            "item_type": "feature_challenger",
            "league": "M+W",
            "depends_on_revised_detailed": False,
            "status": "Probably safe",
            "reason": "Current pace implementation uses compact results only.",
        },
        {
            "item_id": "LATE-FEAT-24",
            "item_type": "feature_challenger",
            "league": "M+W",
            "depends_on_revised_detailed": False,
            "status": "Possibly affected",
            "reason": (
                "Only re-test if the implemented late-form path actually consumed "
                "detailed-derived splits in the runner used for evaluation."
            ),
        },
        {
            "item_id": "LATE-FEAT-29",
            "item_type": "feature_challenger",
            "league": "W",
            "depends_on_revised_detailed": False,
            "status": "Probably safe",
            "reason": (
                "Conference percentile rank path is not detailed-boxscore dependent "
                "in its current implementation."
            ),
        },
    ]
    return pd.DataFrame(rows).sort_values(["status", "item_type", "item_id"]).reset_index(drop=True)


def compare_kaggle_vs_espn_discrepancies(
    *,
    league: str,
    regular_original: pd.DataFrame,
    regular_revised: pd.DataFrame,
    tourney_original: pd.DataFrame,
    tourney_revised: pd.DataFrame,
    espn_root: Path,
    team_spellings_path: Path,
) -> pd.DataFrame:
    """Compare original and revised Kaggle detailed stats against matched ESPN boxscores."""
    team_spellings = _read_team_spellings(team_spellings_path)
    frames = []
    for dataset_version, regular_frame, tourney_frame in [
        ("original", regular_original, tourney_original),
        ("revised", regular_revised, tourney_revised),
    ]:
        matched = _load_matched_espn_comparison_rows(
            league=league,
            regular_frame=regular_frame.loc[regular_frame["Season"].isin(SEASONS_2021_2026)].copy(),
            tourney_frame=tourney_frame.loc[tourney_frame["Season"].isin(SEASONS_2021_2026)].copy(),
            espn_root=espn_root,
            team_spellings=team_spellings,
        )
        if matched.empty:
            continue
        frames.append(
            _summarize_espn_discrepancies(
                matched,
                dataset_version=dataset_version,
                league=league,
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "league",
                "dataset_version",
                "stat",
                "rows",
                "kaggle_std",
                "mean_abs_discrepancy",
                "max_abs_discrepancy",
                "significant_discrepancy_rows",
                "significant_discrepancy_rate",
            ]
        )
    return pd.concat(frames, ignore_index=True).sort_values(
        ["league", "stat", "dataset_version"]
    ).reset_index(drop=True)


def _load_matched_espn_comparison_rows(
    *,
    league: str,
    regular_frame: pd.DataFrame,
    tourney_frame: pd.DataFrame,
    espn_root: Path,
    team_spellings: pd.DataFrame,
) -> pd.DataFrame:
    team_id_col = "MTeamID" if league == "M" else "WTeamID"
    league_root = "mens-college-basketball" if league == "M" else "womens-college-basketball"
    mapping = _build_unique_espn_mapping(team_spellings, team_id_col=team_id_col)
    combined = pd.concat([regular_frame, tourney_frame], ignore_index=True).copy()
    key_counts = combined.groupby(
        ["Season", "WTeamID", "LTeamID", "WScore", "LScore"]
    ).size().rename("key_count")
    combined = combined.merge(
        key_counts.reset_index(),
        on=["Season", "WTeamID", "LTeamID", "WScore", "LScore"],
        how="left",
        validate="many_to_one",
    )
    combined = combined.loc[combined["key_count"] == 1].copy()

    frames: list[pd.DataFrame] = []
    agg_map = {
        "PTS": "sum",
        "FG_att": "sum",
        "OREB": "sum",
        "DREB": "sum",
        "AST": "sum",
        "STL": "sum",
        "BLK": "sum",
        "TO": "sum",
        "PF": "sum",
    }
    rename_map = {
        "PTS": "Score",
        "FG_att": "FGA",
        "OREB": "OR",
        "DREB": "DR",
        "AST": "Ast",
        "STL": "Stl",
        "BLK": "Blk",
        "TO": "TO",
        "PF": "PF",
    }
    for season in SEASONS_2021_2026:
        boxscores_path = espn_root / league_root / str(season) / "boxscores.parquet"
        if not boxscores_path.exists():
            continue
        boxscores = pd.read_parquet(boxscores_path)
        aggregated = (
            boxscores.groupby(["gid", "tid"], as_index=False)
            .agg(**{col: (col, fn) for col, fn in agg_map.items()})
            .assign(espn_id_norm=lambda df: df["tid"].map(_normalize_espn_id))
            .merge(mapping, on="espn_id_norm", how="left", validate="many_to_one")
            .dropna(subset=["TeamID"])
            .assign(TeamID=lambda df: df["TeamID"].astype(int))
            .drop(columns=["tid", "espn_id_norm"])
        )
        paired = _pair_espn_games(aggregated)
        matched_gids = _match_espn_games_to_kaggle(
            season=season,
            paired_games=paired,
            regular_season_results=combined.loc[
                combined["Season"] == season,
                ["Season", "WTeamID", "LTeamID", "WScore", "LScore"],
            ],
        )
        if not matched_gids:
            continue
        matched_pairs = paired.loc[paired["left_gid"].isin(matched_gids)].copy()
        season_kaggle = combined.loc[combined["Season"] == season].copy()
        season_kaggle = season_kaggle.drop(columns=["key_count"])
        espn_rows = []
        for _, row in matched_pairs.iterrows():
            left_score = int(row["left_PTS"])
            right_score = int(row["right_PTS"])
            if left_score >= right_score:
                winner_prefix = "left_"
                loser_prefix = "right_"
            else:
                winner_prefix = "right_"
                loser_prefix = "left_"
            espn_row: dict[str, int] = {
                "Season": season,
                "WTeamID": int(row[f"{winner_prefix}TeamID"]),
                "LTeamID": int(row[f"{loser_prefix}TeamID"]),
                "WScore": int(row[f"{winner_prefix}PTS"]),
                "LScore": int(row[f"{loser_prefix}PTS"]),
            }
            for stat_name, out_name in rename_map.items():
                espn_row[f"W{out_name}"] = int(row[f"{winner_prefix}{stat_name}"])
                espn_row[f"L{out_name}"] = int(row[f"{loser_prefix}{stat_name}"])
            espn_rows.append(espn_row)
        espn_frame = pd.DataFrame(espn_rows)
        if espn_frame.empty:
            continue
        merged = season_kaggle.merge(
            espn_frame,
            on=["Season", "WTeamID", "LTeamID", "WScore", "LScore"],
            suffixes=("_kaggle", "_espn"),
            how="inner",
            validate="one_to_one",
        )
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summarize_espn_discrepancies(
    matched: pd.DataFrame,
    *,
    dataset_version: str,
    league: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stat in ESPN_COMPARISON_STATS:
        kaggle_col = f"{stat}_kaggle"
        espn_col = f"{stat}_espn"
        discrepancy = matched[kaggle_col].astype(float) - matched[espn_col].astype(float)
        kaggle_std = float(matched[kaggle_col].astype(float).std(ddof=0))
        if kaggle_std == 0.0:
            scaled = pd.Series(0.0, index=matched.index, dtype=float)
        else:
            scaled = discrepancy.abs() / kaggle_std
        rows.append(
            {
                "league": league,
                "dataset_version": dataset_version,
                "stat": stat,
                "rows": int(len(matched)),
                "kaggle_std": kaggle_std,
                "mean_abs_discrepancy": float(discrepancy.abs().mean()),
                "max_abs_discrepancy": float(discrepancy.abs().max()),
                "significant_discrepancy_rows": int((scaled >= 1.0).sum()),
                "significant_discrepancy_rate": float((scaled >= 1.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _add_game_key(frame: pd.DataFrame) -> pd.DataFrame:
    keyed = frame.copy()
    keyed["low_team"] = keyed[["WTeamID", "LTeamID"]].min(axis=1).astype(int)
    keyed["high_team"] = keyed[["WTeamID", "LTeamID"]].max(axis=1).astype(int)
    return keyed


def _frame_keys(frame: pd.DataFrame) -> list[tuple[int, int, int, int]]:
    return list(frame.itertuples(index=False, name=None))


def _tracked_compare_cols(frame: pd.DataFrame) -> list[str]:
    exclude = {"Season", "DayNum", "low_team", "high_team"}
    return [col for col in frame.columns if col not in exclude]


def _read_team_spellings(path: Path) -> pd.DataFrame:
    for encoding in (None, "cp1252", "latin1"):
        try:
            if encoding is None:
                return pd.read_csv(path)
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode team spellings file: {path}")
