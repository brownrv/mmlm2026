from __future__ import annotations

from typing import cast

import pandas as pd


def build_adjusted_efficiency_features(
    detailed_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
) -> pd.DataFrame:
    """Build season-level offensive/defensive efficiency features.

    Uses a standard possessions proxy:
    possessions = FGA - OR + TO + 0.475 * FTA
    """
    required = {
        "Season",
        "DayNum",
        "WTeamID",
        "LTeamID",
        "WScore",
        "LScore",
        "WFGA",
        "WFTA",
        "WOR",
        "WTO",
        "LFGA",
        "LFTA",
        "LOR",
        "LTO",
    }
    missing = required.difference(detailed_results.columns)
    if missing:
        raise ValueError(f"Detailed results missing required columns: {sorted(missing)}")

    filtered = detailed_results.loc[detailed_results["DayNum"] < day_cutoff].copy()
    rows: list[dict[str, float | int]] = []
    for _, row in filtered.iterrows():
        winner_poss = _estimate_possessions(
            fga=float(row["WFGA"]),
            fta=float(row["WFTA"]),
            oreb=float(row["WOR"]),
            tov=float(row["WTO"]),
        )
        loser_poss = _estimate_possessions(
            fga=float(row["LFGA"]),
            fta=float(row["LFTA"]),
            oreb=float(row["LOR"]),
            tov=float(row["LTO"]),
        )
        game_possessions = max(1.0, 0.5 * (winner_poss + loser_poss))

        rows.append(
            {
                "Season": int(row["Season"]),
                "TeamID": int(row["WTeamID"]),
                "points_for": float(row["WScore"]),
                "points_against": float(row["LScore"]),
                "game_possessions": game_possessions,
            }
        )
        rows.append(
            {
                "Season": int(row["Season"]),
                "TeamID": int(row["LTeamID"]),
                "points_for": float(row["LScore"]),
                "points_against": float(row["WScore"]),
                "game_possessions": game_possessions,
            }
        )

    season_team = pd.DataFrame(rows)
    aggregated = (
        season_team.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            games=("points_for", "size"),
            points_for=("points_for", "sum"),
            points_against=("points_against", "sum"),
            possessions=("game_possessions", "sum"),
            tempo=("game_possessions", "mean"),
        )
        .assign(
            off_eff=lambda df: 100.0 * df["points_for"] / df["possessions"],
            def_eff=lambda df: 100.0 * df["points_against"] / df["possessions"],
        )
    )
    return aggregated[["Season", "TeamID", "games", "tempo", "off_eff", "def_eff"]]


def build_margin_per_100_features(
    detailed_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
) -> pd.DataFrame:
    """Build season-level margin-per-100-possessions features."""
    team_games = _build_detailed_team_game_rows(detailed_results, day_cutoff=day_cutoff)
    return (
        team_games.groupby(["Season", "TeamID"], as_index=False)
        .agg(mov_per100=("mov_per100", "mean"))
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )


def build_assist_rate_features(
    detailed_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
) -> pd.DataFrame:
    """Build season-level assist-rate features."""
    team_games = _build_detailed_team_game_rows(detailed_results, day_cutoff=day_cutoff)
    aggregated = (
        team_games.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            ast=("ast", "sum"),
            fgm=("fgm", "sum"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )
    aggregated["ast_rate"] = aggregated["ast"] / aggregated["fgm"].clip(lower=1.0)
    return aggregated[["Season", "TeamID", "ast_rate"]]


def build_free_throw_rate_features(
    detailed_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
) -> pd.DataFrame:
    """Build season-level free-throw-rate features using FTM/FGA."""
    team_games = _build_detailed_team_game_rows(detailed_results, day_cutoff=day_cutoff)
    aggregated = (
        team_games.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            ftm=("ftm", "sum"),
            fga=("fga", "sum"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )
    aggregated["ftr"] = aggregated["ftm"] / aggregated["fga"].clip(lower=1.0)
    return aggregated[["Season", "TeamID", "ftr"]]


def build_turnover_rate_features(
    detailed_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
) -> pd.DataFrame:
    """Build season-level turnover-rate features."""
    team_games = _build_detailed_team_game_rows(detailed_results, day_cutoff=day_cutoff)
    aggregated = (
        team_games.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            tov=("tov", "sum"),
            possessions=("possessions", "sum"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )
    aggregated["tov_rate"] = aggregated["tov"] / aggregated["possessions"].clip(lower=1.0)
    return aggregated[["Season", "TeamID", "tov_rate"]]


def build_conf_tourney_win_rate_features(conference_tourney_games: pd.DataFrame) -> pd.DataFrame:
    """Build season-level conference-tournament win rate features."""
    required = {"Season", "WTeamID", "LTeamID"}
    missing = required.difference(conference_tourney_games.columns)
    if missing:
        raise ValueError(f"Conference tourney games missing required columns: {sorted(missing)}")

    rows: list[dict[str, int]] = []
    for _, row in conference_tourney_games.iterrows():
        season = int(row["Season"])
        rows.append({"Season": season, "TeamID": int(row["WTeamID"]), "conf_win": 1})
        rows.append({"Season": season, "TeamID": int(row["LTeamID"]), "conf_win": 0})

    team_games = pd.DataFrame(rows)
    return (
        team_games.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            conf_tourney_games=("conf_win", "size"),
            conf_tourney_win_pct=("conf_win", "mean"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )


def build_strength_of_schedule_features(
    regular_season_results: pd.DataFrame,
    elo_ratings: pd.DataFrame,
    *,
    day_cutoff: int = 134,
) -> pd.DataFrame:
    """Build season-level strength-of-schedule features from opponent end-of-season Elo."""
    required_results = {"Season", "DayNum", "WTeamID", "LTeamID"}
    required_elo = {"Season", "TeamID", "elo"}
    missing_results = required_results.difference(regular_season_results.columns)
    missing_elo = required_elo.difference(elo_ratings.columns)
    if missing_results:
        raise ValueError(
            f"Regular season results missing required columns: {sorted(missing_results)}"
        )
    if missing_elo:
        raise ValueError(f"Elo ratings missing required columns: {sorted(missing_elo)}")

    filtered = regular_season_results.loc[regular_season_results["DayNum"] < day_cutoff].copy()
    rows: list[dict[str, int]] = []
    for _, row in filtered.iterrows():
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        rows.append({"Season": season, "TeamID": winner, "OpponentTeamID": loser})
        rows.append({"Season": season, "TeamID": loser, "OpponentTeamID": winner})

    games = pd.DataFrame(rows)
    elo = elo_ratings.rename(columns={"TeamID": "OpponentTeamID", "elo": "opponent_elo"})
    merged = games.merge(
        elo,
        on=["Season", "OpponentTeamID"],
        how="left",
        validate="many_to_one",
    )
    if merged["opponent_elo"].isna().any():
        raise ValueError(
            "Missing opponent Elo values while building strength-of-schedule features."
        )

    aggregated = (
        merged.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            games=("opponent_elo", "size"),
            sos=("opponent_elo", "mean"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )
    return aggregated


def build_recent_form_features(
    regular_season_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
    last_n_games: int = 10,
) -> pd.DataFrame:
    """Build season-level recent-form features from the last N pre-cutoff games."""
    required = {"Season", "DayNum", "WTeamID", "LTeamID"}
    missing = required.difference(regular_season_results.columns)
    if missing:
        raise ValueError(f"Regular season results missing required columns: {sorted(missing)}")

    filtered = regular_season_results.loc[regular_season_results["DayNum"] < day_cutoff].copy()
    rows: list[dict[str, int]] = []
    for _, row in filtered.iterrows():
        season = int(row["Season"])
        day_num = int(row["DayNum"])
        rows.append({"Season": season, "DayNum": day_num, "TeamID": int(row["WTeamID"]), "win": 1})
        rows.append({"Season": season, "DayNum": day_num, "TeamID": int(row["LTeamID"]), "win": 0})

    team_games = (
        pd.DataFrame(rows).sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)
    )
    recent = (
        team_games.groupby(["Season", "TeamID"], as_index=False)
        .tail(last_n_games)
        .groupby(["Season", "TeamID"], as_index=False)
        .agg(
            recent_games=("win", "size"),
            recent_win_pct=("win", "mean"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )
    return recent


def build_close_game_win_rate_features(
    regular_season_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
    margin_threshold: int = 3,
) -> pd.DataFrame:
    """Build season-level close-game win rate features.

    Teams with no close games receive `close_games = 0` and `close_win_pct = 0.0`.
    """
    required = {"Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore"}
    missing = required.difference(regular_season_results.columns)
    if missing:
        raise ValueError(f"Regular season results missing required columns: {sorted(missing)}")

    filtered = regular_season_results.loc[regular_season_results["DayNum"] < day_cutoff].copy()
    all_team_rows: list[dict[str, int]] = []
    close_rows: list[dict[str, int]] = []
    for _, row in filtered.iterrows():
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        margin = abs(int(row["WScore"]) - int(row["LScore"]))
        all_team_rows.append({"Season": season, "TeamID": winner})
        all_team_rows.append({"Season": season, "TeamID": loser})
        if margin <= margin_threshold:
            close_rows.append({"Season": season, "TeamID": winner, "close_win": 1})
            close_rows.append({"Season": season, "TeamID": loser, "close_win": 0})

    teams = pd.DataFrame(all_team_rows).drop_duplicates().sort_values(["Season", "TeamID"])
    if close_rows:
        close_games = pd.DataFrame(close_rows)
        aggregated = (
            close_games.groupby(["Season", "TeamID"], as_index=False)
            .agg(
                close_games=("close_win", "size"),
                close_win_pct=("close_win", "mean"),
            )
            .sort_values(["Season", "TeamID"])
            .reset_index(drop=True)
        )
    else:
        aggregated = pd.DataFrame(columns=["Season", "TeamID", "close_games", "close_win_pct"])

    merged = teams.merge(
        aggregated,
        on=["Season", "TeamID"],
        how="left",
        validate="one_to_one",
    )
    if "close_games" not in merged.columns:
        merged["close_games"] = 0
    else:
        merged["close_games"] = merged["close_games"].where(
            merged["close_games"].notna(),
            0,
        )
    if "close_win_pct" not in merged.columns:
        merged["close_win_pct"] = 0.0
    else:
        merged["close_win_pct"] = merged["close_win_pct"].where(
            merged["close_win_pct"].notna(),
            0.0,
        )
    merged["close_games"] = merged["close_games"].astype(int)
    merged["close_win_pct"] = merged["close_win_pct"].astype(float)
    return merged.reset_index(drop=True)


def build_massey_consensus_features(
    massey_ordinals: pd.DataFrame,
    *,
    ranking_day_cutoff: int = 133,
) -> pd.DataFrame:
    """Build season-level median Massey rank from the last pre-cutoff ranking per system."""
    required = {"Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"}
    missing = required.difference(massey_ordinals.columns)
    if missing:
        raise ValueError(f"Massey ordinals missing required columns: {sorted(missing)}")

    filtered = massey_ordinals.loc[massey_ordinals["RankingDayNum"] <= ranking_day_cutoff].copy()
    latest_by_system = (
        filtered.sort_values(["Season", "TeamID", "SystemName", "RankingDayNum"])
        .groupby(["Season", "TeamID", "SystemName"], as_index=False)
        .tail(1)
    )
    aggregated = (
        latest_by_system.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            massey_system_count=("SystemName", "nunique"),
            massey_median_rank=("OrdinalRank", "median"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )
    return aggregated


def build_schedule_adjusted_net_eff_features(
    efficiency_features: pd.DataFrame,
    sos_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build a season-normalized combined strength index from net efficiency and SOS.

    The index is defined within each season as:
    `sos_adjusted_net_eff = z(net_eff) + z(sos)`,
    where `net_eff = off_eff - def_eff`.
    """
    required_efficiency = {"Season", "TeamID", "off_eff", "def_eff"}
    required_sos = {"Season", "TeamID", "sos"}
    missing_efficiency = required_efficiency.difference(efficiency_features.columns)
    missing_sos = required_sos.difference(sos_features.columns)
    if missing_efficiency:
        raise ValueError(
            f"Efficiency features missing required columns: {sorted(missing_efficiency)}"
        )
    if missing_sos:
        raise ValueError(f"SOS features missing required columns: {sorted(missing_sos)}")

    merged = efficiency_features.merge(
        sos_features[["Season", "TeamID", "sos"]],
        on=["Season", "TeamID"],
        how="inner",
        validate="one_to_one",
    ).copy()
    merged["net_eff"] = merged["off_eff"] - merged["def_eff"]

    rows: list[pd.DataFrame] = []
    for _season, season_frame in merged.groupby("Season", sort=True):
        frame = season_frame.copy()
        frame["net_eff_z"] = _safe_zscore(frame["net_eff"])
        frame["sos_z"] = _safe_zscore(frame["sos"])
        frame["sos_adjusted_net_eff"] = frame["net_eff_z"] + frame["sos_z"]
        rows.append(frame[["Season", "TeamID", "net_eff", "sos_adjusted_net_eff"]])

    return (
        pd.concat(rows, ignore_index=True).sort_values(["Season", "TeamID"]).reset_index(drop=True)
    )


def build_iterative_adjusted_efficiency_features(
    detailed_results: pd.DataFrame,
    *,
    day_cutoff: int = 134,
    n_iter: int = 100,
    tol: float = 1e-4,
    women_hca: float = 0.0,
) -> pd.DataFrame:
    """Build KenPom-style iterative opponent-adjusted efficiency features.

    This is a challenger implementation aligned to the `adj_quality_gap_v10`
    documentation. It uses a Dean Oliver-style possessions estimate with a
    0.44 free-throw coefficient and iteratively reweights each game's raw
    offensive and defensive efficiency by opponent quality.
    """
    required = {
        "Season",
        "DayNum",
        "WTeamID",
        "LTeamID",
        "WScore",
        "LScore",
        "WFGA",
        "WFTA",
        "WOR",
        "WTO",
        "LFGA",
        "LFTA",
        "LOR",
        "LTO",
    }
    if women_hca != 0.0:
        required.add("WLoc")
    missing = required.difference(detailed_results.columns)
    if missing:
        raise ValueError(f"Detailed results missing required columns: {sorted(missing)}")

    filtered = detailed_results.loc[detailed_results["DayNum"] < day_cutoff].copy()
    game_rows: list[dict[str, float | int]] = []
    for _, row in filtered.iterrows():
        winner_poss = _estimate_possessions_challenger(
            fga=float(row["WFGA"]),
            fta=float(row["WFTA"]),
            oreb=float(row["WOR"]),
            tov=float(row["WTO"]),
        )
        loser_poss = _estimate_possessions_challenger(
            fga=float(row["LFGA"]),
            fta=float(row["LFTA"]),
            oreb=float(row["LOR"]),
            tov=float(row["LTO"]),
        )
        possessions = max(1.0, 0.5 * (winner_poss + loser_poss))
        winner_score, loser_score = _apply_women_hca_score_adjustment(
            winner_score=float(row["WScore"]),
            loser_score=float(row["LScore"]),
            winner_location=str(row["WLoc"]) if "WLoc" in row.index else "N",
            women_hca=women_hca,
        )
        winner_off = 100.0 * winner_score / possessions
        winner_def = 100.0 * loser_score / possessions
        loser_off = 100.0 * loser_score / possessions
        loser_def = 100.0 * winner_score / possessions

        game_rows.append(
            {
                "Season": int(row["Season"]),
                "TeamID": int(row["WTeamID"]),
                "OpponentTeamID": int(row["LTeamID"]),
                "off_g": winner_off,
                "def_g": winner_def,
            }
        )
        game_rows.append(
            {
                "Season": int(row["Season"]),
                "TeamID": int(row["LTeamID"]),
                "OpponentTeamID": int(row["WTeamID"]),
                "off_g": loser_off,
                "def_g": loser_def,
            }
        )

    game_frame = pd.DataFrame(game_rows)
    season_outputs: list[pd.DataFrame] = []
    for season, season_games in game_frame.groupby("Season", sort=True):
        team_means = (
            season_games.groupby("TeamID", as_index=False)
            .agg(
                raw_off_eff=("off_g", "mean"),
                raw_def_eff=("def_g", "mean"),
                games=("off_g", "size"),
            )
            .sort_values("TeamID")
            .reset_index(drop=True)
        )
        ratings = team_means.copy()
        ratings["adj_off_eff"] = ratings["raw_off_eff"]
        ratings["adj_def_eff"] = ratings["raw_def_eff"]
        previous_adj_off = ratings["adj_off_eff"].copy()

        for _ in range(n_iter):
            league_avg_off = float(ratings["adj_off_eff"].mean())
            league_avg_def = float(ratings["adj_def_eff"].mean())
            opponent_lookup = ratings.rename(
                columns={
                    "TeamID": "OpponentTeamID",
                    "adj_off_eff": "opp_adj_off_eff",
                    "adj_def_eff": "opp_adj_def_eff",
                }
            )[["OpponentTeamID", "opp_adj_off_eff", "opp_adj_def_eff"]]
            adjusted_games = season_games.merge(
                opponent_lookup,
                on="OpponentTeamID",
                how="left",
                validate="many_to_one",
            )
            adjusted_games["adj_off_game"] = (
                adjusted_games["off_g"] / adjusted_games["opp_adj_def_eff"] * league_avg_def
            )
            adjusted_games["adj_def_game"] = (
                adjusted_games["def_g"] / adjusted_games["opp_adj_off_eff"] * league_avg_off
            )
            updated = (
                adjusted_games.groupby("TeamID", as_index=False)
                .agg(
                    adj_off_eff=("adj_off_game", "mean"),
                    adj_def_eff=("adj_def_game", "mean"),
                )
                .sort_values("TeamID")
                .reset_index(drop=True)
            )
            mean_delta = float((updated["adj_off_eff"] - previous_adj_off).abs().mean())
            ratings = ratings.drop(columns=["adj_off_eff", "adj_def_eff"]).merge(
                updated,
                on="TeamID",
                how="left",
                validate="one_to_one",
            )
            previous_adj_off = ratings["adj_off_eff"].copy()
            if mean_delta < tol:
                break

        season_output = ratings.copy()
        season_output.insert(0, "Season", cast(int, season))
        season_output["adj_net_eff"] = season_output["adj_off_eff"] - season_output["adj_def_eff"]
        season_outputs.append(
            season_output[
                [
                    "Season",
                    "TeamID",
                    "games",
                    "raw_off_eff",
                    "raw_def_eff",
                    "adj_off_eff",
                    "adj_def_eff",
                    "adj_net_eff",
                ]
            ]
        )

    return (
        pd.concat(season_outputs, ignore_index=True)
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )


def _estimate_possessions(*, fga: float, fta: float, oreb: float, tov: float) -> float:
    return fga - oreb + tov + 0.475 * fta


def _build_detailed_team_game_rows(
    detailed_results: pd.DataFrame,
    *,
    day_cutoff: int,
) -> pd.DataFrame:
    core_required = {
        "Season",
        "DayNum",
        "WTeamID",
        "LTeamID",
        "WScore",
        "LScore",
        "WFGA",
        "WFTA",
        "WOR",
        "WTO",
        "LFGA",
        "LFTA",
        "LOR",
        "LTO",
    }
    missing = core_required.difference(detailed_results.columns)
    if missing:
        raise ValueError(f"Detailed results missing required columns: {sorted(missing)}")

    filtered = detailed_results.loc[detailed_results["DayNum"] < day_cutoff].copy()
    # Some lightweight fixtures omit made-shot and assist columns. Fall back to
    # conservative approximations so the feature builders remain usable.
    for prefix in ("W", "L"):
        fgm_col = f"{prefix}FGM"
        fga_col = f"{prefix}FGA"
        ftm_col = f"{prefix}FTM"
        fta_col = f"{prefix}FTA"
        ast_col = f"{prefix}Ast"
        if fgm_col not in filtered.columns:
            filtered[fgm_col] = filtered[fga_col]
        if ftm_col not in filtered.columns:
            filtered[ftm_col] = filtered[fta_col]
        if ast_col not in filtered.columns:
            filtered[ast_col] = filtered[fgm_col]

    rows: list[dict[str, float | int]] = []
    for _, row in filtered.iterrows():
        winner_poss = _estimate_possessions_challenger(
            fga=float(row["WFGA"]),
            fta=float(row["WFTA"]),
            oreb=float(row["WOR"]),
            tov=float(row["WTO"]),
        )
        loser_poss = _estimate_possessions_challenger(
            fga=float(row["LFGA"]),
            fta=float(row["LFTA"]),
            oreb=float(row["LOR"]),
            tov=float(row["LTO"]),
        )
        possessions = max(1.0, 0.5 * (winner_poss + loser_poss))
        mov_per100 = 100.0 * (float(row["WScore"]) - float(row["LScore"])) / possessions
        rows.append(
            {
                "Season": int(row["Season"]),
                "TeamID": int(row["WTeamID"]),
                "possessions": possessions,
                "mov_per100": mov_per100,
                "fgm": float(row["WFGM"]),
                "fga": float(row["WFGA"]),
                "ftm": float(row["WFTM"]),
                "ast": float(row["WAst"]),
                "tov": float(row["WTO"]),
            }
        )
        rows.append(
            {
                "Season": int(row["Season"]),
                "TeamID": int(row["LTeamID"]),
                "possessions": possessions,
                "mov_per100": -mov_per100,
                "fgm": float(row["LFGM"]),
                "fga": float(row["LFGA"]),
                "ftm": float(row["LFTM"]),
                "ast": float(row["LAst"]),
                "tov": float(row["LTO"]),
            }
        )

    return pd.DataFrame(rows)


def _safe_zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0.0:
        return pd.Series([0.0] * len(series), index=series.index, dtype=float)
    mean = float(series.mean())
    return (series.astype(float) - mean) / std


def _estimate_possessions_challenger(*, fga: float, fta: float, oreb: float, tov: float) -> float:
    return fga - oreb + tov + 0.44 * fta


def _apply_women_hca_score_adjustment(
    *,
    winner_score: float,
    loser_score: float,
    winner_location: str,
    women_hca: float,
) -> tuple[float, float]:
    if women_hca == 0.0:
        return winner_score, loser_score

    half_adjustment = women_hca / 2.0
    if winner_location == "H":
        return winner_score - half_adjustment, loser_score + half_adjustment
    if winner_location == "A":
        return winner_score + half_adjustment, loser_score - half_adjustment
    return winner_score, loser_score
