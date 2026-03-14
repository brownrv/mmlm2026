from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd


def build_espn_women_four_factor_strength_features(
    *,
    season: int,
    boxscores: pd.DataFrame,
    regular_season_results: pd.DataFrame,
    team_spellings: pd.DataFrame,
) -> pd.DataFrame:
    """Build season-level women four-factor strength features from ESPN boxscores."""
    return _build_espn_four_factor_strength_features(
        season=season,
        boxscores=boxscores,
        regular_season_results=regular_season_results,
        team_spellings=team_spellings,
        team_id_col="WTeamID",
    )


def build_espn_men_four_factor_strength_features(
    *,
    season: int,
    boxscores: pd.DataFrame,
    regular_season_results: pd.DataFrame,
    team_spellings: pd.DataFrame,
) -> pd.DataFrame:
    """Build season-level men four-factor strength features from ESPN boxscores."""
    return _build_espn_four_factor_strength_features(
        season=season,
        boxscores=boxscores,
        regular_season_results=regular_season_results,
        team_spellings=team_spellings,
        team_id_col="MTeamID",
    )


def _build_espn_four_factor_strength_features(
    *,
    season: int,
    boxscores: pd.DataFrame,
    regular_season_results: pd.DataFrame,
    team_spellings: pd.DataFrame,
    team_id_col: str,
) -> pd.DataFrame:
    """Build season-level four-factor strength features from ESPN boxscores.

    ESPN games are kept only if they can be matched cleanly to Kaggle regular-season
    results for the same season. This avoids using postseason-only rows.
    """
    required_box = {
        "gid",
        "tid",
        "FG_made",
        "FG_att",
        "3PT_made",
        "FT_att",
        "OREB",
        "DREB",
        "TO",
        "PTS",
    }
    missing_box = required_box.difference(boxscores.columns)
    if missing_box:
        raise ValueError(f"ESPN boxscores missing required columns: {sorted(missing_box)}")

    required_results = {"Season", "WTeamID", "LTeamID", "WScore", "LScore"}
    missing_results = required_results.difference(regular_season_results.columns)
    if missing_results:
        raise ValueError(
            f"Regular season results missing required columns: {sorted(missing_results)}"
        )

    mapping = _build_unique_espn_mapping(team_spellings, team_id_col=team_id_col)
    aggregated = (
        boxscores.groupby(["gid", "tid"], as_index=False)
        .agg(
            FG_made=("FG_made", "sum"),
            FG_att=("FG_att", "sum"),
            threes_made=("3PT_made", "sum"),
            FT_att=("FT_att", "sum"),
            OREB=("OREB", "sum"),
            DREB=("DREB", "sum"),
            TO=("TO", "sum"),
            PTS=("PTS", "sum"),
        )
        .assign(espn_id_norm=lambda df: df["tid"].map(_normalize_espn_id))
        .merge(mapping, on="espn_id_norm", how="left", validate="many_to_one")
        .dropna(subset=["TeamID"])
        .assign(TeamID=lambda df: df["TeamID"].astype(int))
    )

    paired = _pair_espn_games(aggregated)
    matched_gids = _match_espn_games_to_kaggle(
        season=season,
        paired_games=paired,
        regular_season_results=regular_season_results,
    )
    if not matched_gids:
        return pd.DataFrame(
            columns=[
                "Season",
                "TeamID",
                "espn_efg",
                "espn_tov_rate",
                "espn_orb_pct",
                "espn_ftr",
                "espn_opp_efg",
                "espn_opp_tov_rate",
                "espn_opp_orb_pct",
                "espn_opp_ftr",
                "espn_four_factor_strength",
            ]
        )

    matched = aggregated.loc[aggregated["gid"].isin(matched_gids)].copy()
    opponent = matched.rename(
        columns={
            "TeamID": "OpponentTeamID",
            "FG_made": "opp_fg_made",
            "FG_att": "opp_fg_att",
            "threes_made": "opp_threes_made",
            "FT_att": "opp_ft_att",
            "OREB": "opp_oreb",
            "DREB": "opp_dreb",
            "TO": "opp_to",
            "PTS": "opp_pts",
        }
    )[
        [
            "gid",
            "OpponentTeamID",
            "opp_fg_made",
            "opp_fg_att",
            "opp_threes_made",
            "opp_ft_att",
            "opp_oreb",
            "opp_dreb",
            "opp_to",
            "opp_pts",
        ]
    ]
    team_games = (
        matched.merge(opponent, on="gid", how="left")
        .loc[lambda df: df["TeamID"] != df["OpponentTeamID"]]
        .copy()
    )

    aggregated_team = (
        team_games.groupby("TeamID", as_index=False)
        .agg(
            fg_made=("FG_made", "sum"),
            fg_att=("FG_att", "sum"),
            threes_made=("threes_made", "sum"),
            ft_att=("FT_att", "sum"),
            oreb=("OREB", "sum"),
            dreb=("DREB", "sum"),
            tov=("TO", "sum"),
            pts=("PTS", "sum"),
            opp_fg_made=("opp_fg_made", "sum"),
            opp_fg_att=("opp_fg_att", "sum"),
            opp_threes_made=("opp_threes_made", "sum"),
            opp_ft_att=("opp_ft_att", "sum"),
            opp_oreb=("opp_oreb", "sum"),
            opp_dreb=("opp_dreb", "sum"),
            opp_to=("opp_to", "sum"),
            opp_pts=("opp_pts", "sum"),
        )
        .sort_values("TeamID")
        .reset_index(drop=True)
    )

    aggregated_team["espn_efg"] = (
        aggregated_team["fg_made"] + 0.5 * aggregated_team["threes_made"]
    ) / aggregated_team["fg_att"].clip(lower=1.0)
    possessions = (
        aggregated_team["fg_att"]
        - aggregated_team["oreb"]
        + aggregated_team["tov"]
        + 0.475 * aggregated_team["ft_att"]
    )
    aggregated_team["espn_tov_rate"] = aggregated_team["tov"] / possessions.clip(lower=1.0)
    aggregated_team["espn_orb_pct"] = aggregated_team["oreb"] / (
        aggregated_team["oreb"] + aggregated_team["opp_dreb"]
    ).clip(lower=1.0)
    aggregated_team["espn_ftr"] = aggregated_team["ft_att"] / aggregated_team["fg_att"].clip(
        lower=1.0
    )

    aggregated_team["espn_opp_efg"] = (
        aggregated_team["opp_fg_made"] + 0.5 * aggregated_team["opp_threes_made"]
    ) / aggregated_team["opp_fg_att"].clip(lower=1.0)
    opp_possessions = (
        aggregated_team["opp_fg_att"]
        - aggregated_team["opp_oreb"]
        + aggregated_team["opp_to"]
        + 0.475 * aggregated_team["opp_ft_att"]
    )
    aggregated_team["espn_opp_tov_rate"] = aggregated_team["opp_to"] / opp_possessions.clip(
        lower=1.0
    )
    aggregated_team["espn_opp_orb_pct"] = aggregated_team["opp_oreb"] / (
        aggregated_team["opp_oreb"] + aggregated_team["dreb"]
    ).clip(lower=1.0)
    aggregated_team["espn_opp_ftr"] = aggregated_team["opp_ft_att"] / aggregated_team[
        "opp_fg_att"
    ].clip(lower=1.0)

    for column in [
        "espn_efg",
        "espn_tov_rate",
        "espn_orb_pct",
        "espn_ftr",
        "espn_opp_efg",
        "espn_opp_tov_rate",
        "espn_opp_orb_pct",
        "espn_opp_ftr",
    ]:
        aggregated_team[f"{column}_z"] = _safe_zscore(aggregated_team[column])

    aggregated_team["espn_four_factor_strength"] = (
        0.4 * aggregated_team["espn_efg_z"]
        - 0.25 * aggregated_team["espn_tov_rate_z"]
        + 0.2 * aggregated_team["espn_orb_pct_z"]
        + 0.15 * aggregated_team["espn_ftr_z"]
        - 0.4 * aggregated_team["espn_opp_efg_z"]
        + 0.25 * aggregated_team["espn_opp_tov_rate_z"]
        - 0.2 * aggregated_team["espn_opp_orb_pct_z"]
        - 0.15 * aggregated_team["espn_opp_ftr_z"]
    )

    return aggregated_team[
        [
            "TeamID",
            "espn_efg",
            "espn_tov_rate",
            "espn_orb_pct",
            "espn_ftr",
            "espn_opp_efg",
            "espn_opp_tov_rate",
            "espn_opp_orb_pct",
            "espn_opp_ftr",
            "espn_four_factor_strength",
        ]
    ].assign(Season=season)[
        [
            "Season",
            "TeamID",
            "espn_efg",
            "espn_tov_rate",
            "espn_orb_pct",
            "espn_ftr",
            "espn_opp_efg",
            "espn_opp_tov_rate",
            "espn_opp_orb_pct",
            "espn_opp_ftr",
            "espn_four_factor_strength",
        ]
    ]


def load_espn_men_four_factor_strength_features(
    *,
    espn_root: Path,
    seasons: list[int],
    regular_season_results: pd.DataFrame,
    team_spellings_path: Path,
) -> pd.DataFrame:
    """Load and aggregate ESPN-derived men four-factor strength features across seasons."""
    team_spellings = pd.read_csv(team_spellings_path, encoding="latin1")
    frames: list[pd.DataFrame] = []
    for season in seasons:
        boxscores_path = espn_root / str(season) / "boxscores.parquet"
        if not boxscores_path.exists():
            continue
        boxscores = pd.read_parquet(boxscores_path)
        season_results = regular_season_results.loc[
            regular_season_results["Season"] == season
        ].copy()
        if season_results.empty:
            continue
        season_features = build_espn_men_four_factor_strength_features(
            season=season,
            boxscores=boxscores,
            regular_season_results=season_results,
            team_spellings=team_spellings,
        )
        if not season_features.empty:
            frames.append(season_features)

    if not frames:
        return pd.DataFrame(
            columns=[
                "Season",
                "TeamID",
                "espn_efg",
                "espn_tov_rate",
                "espn_orb_pct",
                "espn_ftr",
                "espn_opp_efg",
                "espn_opp_tov_rate",
                "espn_opp_orb_pct",
                "espn_opp_ftr",
                "espn_four_factor_strength",
            ]
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )


def load_espn_women_four_factor_strength_features(
    *,
    espn_root: Path,
    seasons: list[int],
    regular_season_results: pd.DataFrame,
    team_spellings_path: Path,
) -> pd.DataFrame:
    """Load and aggregate ESPN-derived women four-factor strength features."""
    team_spellings = pd.read_csv(team_spellings_path, encoding="latin1")
    frames: list[pd.DataFrame] = []
    for season in seasons:
        boxscores_path = espn_root / str(season) / "boxscores.parquet"
        if not boxscores_path.exists():
            continue
        boxscores = pd.read_parquet(boxscores_path)
        season_results = regular_season_results.loc[
            regular_season_results["Season"] == season
        ].copy()
        if season_results.empty:
            continue
        season_features = build_espn_women_four_factor_strength_features(
            season=season,
            boxscores=boxscores,
            regular_season_results=season_results,
            team_spellings=team_spellings,
        )
        if not season_features.empty:
            frames.append(season_features)

    if not frames:
        return pd.DataFrame(
            columns=[
                "Season",
                "TeamID",
                "espn_efg",
                "espn_tov_rate",
                "espn_orb_pct",
                "espn_ftr",
                "espn_opp_efg",
                "espn_opp_tov_rate",
                "espn_opp_orb_pct",
                "espn_opp_ftr",
                "espn_four_factor_strength",
            ]
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )


def build_espn_men_rotation_stability_features(
    *,
    season: int,
    boxscores: pd.DataFrame,
    regular_season_results: pd.DataFrame,
    team_spellings: pd.DataFrame,
) -> pd.DataFrame:
    """Build season-level men rotation-stability features from ESPN boxscores."""
    required_box = {"gid", "tid", "aid", "starterBench", "MIN"}
    missing_box = required_box.difference(boxscores.columns)
    if missing_box:
        raise ValueError(f"ESPN boxscores missing required columns: {sorted(missing_box)}")

    mapping = _build_unique_espn_mapping(team_spellings, team_id_col="MTeamID")
    mapped_rows = (
        boxscores.assign(espn_id_norm=lambda df: df["tid"].map(_normalize_espn_id))
        .merge(mapping, on="espn_id_norm", how="left", validate="many_to_one")
        .dropna(subset=["TeamID", "aid"])
        .assign(
            TeamID=lambda df: df["TeamID"].astype(int),
            aid=lambda df: df["aid"].astype(str),
            MIN=lambda df: pd.to_numeric(df["MIN"], errors="coerce").fillna(0.0).astype(float),
        )
    )

    aggregated = (
        mapped_rows.groupby(["gid", "TeamID"], as_index=False)
        .agg(PTS=("PTS", "sum"))
        .sort_values(["gid", "TeamID"])
        .reset_index(drop=True)
    )
    paired = _pair_espn_games(aggregated)
    matched_gids = _match_espn_games_to_kaggle(
        season=season,
        paired_games=paired,
        regular_season_results=regular_season_results,
    )
    if not matched_gids:
        return pd.DataFrame(
            columns=[
                "Season",
                "TeamID",
                "espn_top5_minutes_share",
                "espn_top_player_minutes_share",
                "espn_starter_lineup_stability",
                "espn_rotation_stability",
            ]
        )

    matched_rows = mapped_rows.loc[mapped_rows["gid"].isin(matched_gids)].copy()
    total_minutes = (
        matched_rows.groupby(["TeamID"], as_index=False)
        .agg(total_minutes=("MIN", "sum"))
        .sort_values("TeamID")
    )
    player_minutes = (
        matched_rows.groupby(["TeamID", "aid"], as_index=False)
        .agg(player_minutes=("MIN", "sum"))
        .sort_values(["TeamID", "player_minutes"], ascending=[True, False])
    )
    top5 = (
        player_minutes.groupby("TeamID", group_keys=False)
        .head(5)
        .groupby("TeamID", as_index=False)
        .agg(top5_minutes=("player_minutes", "sum"))
    )
    top1 = player_minutes.groupby("TeamID", as_index=False).agg(
        top_player_minutes=("player_minutes", "max")
    )

    starters = matched_rows.loc[matched_rows["starterBench"] == "starters"].copy()
    if starters.empty:
        lineup_stability = pd.DataFrame(
            {
                "TeamID": total_minutes["TeamID"],
                "espn_starter_lineup_stability": 0.0,
            }
        )
    else:
        lineup_keys = (
            starters.groupby(["TeamID", "gid"])["aid"]
            .apply(lambda values: "|".join(sorted(values.astype(str).tolist())))
            .reset_index(name="lineup_key")
        )
        lineup_stability = (
            lineup_keys.groupby("TeamID")["lineup_key"]
            .value_counts(normalize=True)
            .groupby("TeamID")
            .max()
            .reset_index(name="espn_starter_lineup_stability")
        )

    features = (
        total_minutes.merge(top5, on="TeamID", how="left")
        .merge(top1, on="TeamID", how="left")
        .merge(lineup_stability, on="TeamID", how="left")
        .fillna(
            {
                "top5_minutes": 0.0,
                "top_player_minutes": 0.0,
                "espn_starter_lineup_stability": 0.0,
            }
        )
        .sort_values("TeamID")
        .reset_index(drop=True)
    )
    features["espn_top5_minutes_share"] = features["top5_minutes"] / features["total_minutes"].clip(
        lower=1.0
    )
    features["espn_top_player_minutes_share"] = features["top_player_minutes"] / features[
        "total_minutes"
    ].clip(lower=1.0)
    features["espn_rotation_stability"] = (
        _safe_zscore(features["espn_top5_minutes_share"])
        + _safe_zscore(features["espn_starter_lineup_stability"])
        - _safe_zscore(features["espn_top_player_minutes_share"])
    )
    return features[
        [
            "TeamID",
            "espn_top5_minutes_share",
            "espn_top_player_minutes_share",
            "espn_starter_lineup_stability",
            "espn_rotation_stability",
        ]
    ].assign(Season=season)[
        [
            "Season",
            "TeamID",
            "espn_top5_minutes_share",
            "espn_top_player_minutes_share",
            "espn_starter_lineup_stability",
            "espn_rotation_stability",
        ]
    ]


def load_espn_men_rotation_stability_features(
    *,
    espn_root: Path,
    seasons: list[int],
    regular_season_results: pd.DataFrame,
    team_spellings_path: Path,
) -> pd.DataFrame:
    """Load season-level men rotation-stability features across seasons."""
    team_spellings = pd.read_csv(team_spellings_path, encoding="latin1")
    frames: list[pd.DataFrame] = []
    for season in seasons:
        boxscores_path = espn_root / str(season) / "boxscores.parquet"
        if not boxscores_path.exists():
            continue
        boxscores = pd.read_parquet(boxscores_path)
        season_results = regular_season_results.loc[
            regular_season_results["Season"] == season
        ].copy()
        if season_results.empty:
            continue
        season_features = build_espn_men_rotation_stability_features(
            season=season,
            boxscores=boxscores,
            regular_season_results=season_results,
            team_spellings=team_spellings,
        )
        if not season_features.empty:
            frames.append(season_features)

    if not frames:
        return pd.DataFrame(
            columns=[
                "Season",
                "TeamID",
                "espn_top5_minutes_share",
                "espn_top_player_minutes_share",
                "espn_starter_lineup_stability",
                "espn_rotation_stability",
            ]
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )


def _build_unique_espn_mapping(team_spellings: pd.DataFrame, *, team_id_col: str) -> pd.DataFrame:
    mapping = team_spellings[[team_id_col, "espn_id"]].dropna().copy()
    mapping["espn_id_norm"] = mapping["espn_id"].map(_normalize_espn_id)
    mapping = mapping.dropna(subset=["espn_id_norm"]).copy()
    mapping[team_id_col] = mapping[team_id_col].astype(int)

    team_counts = mapping.groupby("espn_id_norm")[team_id_col].nunique()
    unique_ids = team_counts.loc[team_counts == 1].index
    mapping = mapping.loc[mapping["espn_id_norm"].isin(unique_ids), ["espn_id_norm", team_id_col]]
    return mapping.drop_duplicates().rename(columns={team_id_col: "TeamID"})


def _pair_espn_games(aggregated: pd.DataFrame) -> pd.DataFrame:
    counts = aggregated.groupby("gid")["TeamID"].nunique()
    valid_gids = counts.loc[counts == 2].index
    two_team = aggregated.loc[aggregated["gid"].isin(valid_gids)].copy()
    left = two_team.add_prefix("left_")
    right = two_team.add_prefix("right_")
    paired = left.merge(right, left_on="left_gid", right_on="right_gid", how="inner")
    return paired.loc[paired["left_TeamID"] < paired["right_TeamID"]].copy()


def _match_espn_games_to_kaggle(
    *,
    season: int,
    paired_games: pd.DataFrame,
    regular_season_results: pd.DataFrame,
) -> set[str]:
    normalized_games = paired_games.copy()
    normalized_games["low_team"] = normalized_games["left_TeamID"]
    normalized_games["high_team"] = normalized_games["right_TeamID"]
    normalized_games["low_score"] = normalized_games["left_PTS"]
    normalized_games["high_score"] = normalized_games["right_PTS"]

    kaggle = regular_season_results.copy()
    kaggle["low_team"] = kaggle[["WTeamID", "LTeamID"]].min(axis=1)
    kaggle["high_team"] = kaggle[["WTeamID", "LTeamID"]].max(axis=1)
    kaggle["low_score"] = kaggle.apply(
        lambda row: row["WScore"] if row["WTeamID"] < row["LTeamID"] else row["LScore"],
        axis=1,
    )
    kaggle["high_score"] = kaggle.apply(
        lambda row: row["LScore"] if row["WTeamID"] < row["LTeamID"] else row["WScore"],
        axis=1,
    )

    matched = normalized_games.merge(
        kaggle[["low_team", "high_team", "low_score", "high_score"]],
        on=["low_team", "high_team", "low_score", "high_score"],
        how="inner",
    )
    gid_counts = matched.groupby("left_gid").size()
    return {cast(str, gid) for gid in gid_counts.loc[gid_counts == 1].index}


def _normalize_espn_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _safe_zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0.0 or pd.isna(std):
        return pd.Series(0.0, index=series.index, dtype=float)
    mean = float(series.mean())
    return (series.astype(float) - mean) / std
