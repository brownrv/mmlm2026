from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mmlm2026.evaluation.bracket import BracketDiagnostics, compute_bracket_diagnostics
from mmlm2026.evaluation.validation import build_logistic_pipeline
from mmlm2026.features.elo import (
    build_elo_seed_submission_features,
    build_elo_seed_tourney_features,
    compute_pre_tourney_elo_ratings,
)
from mmlm2026.features.primary import (
    build_phase_ab_submission_features,
    build_phase_ab_team_features,
    build_phase_ab_tourney_features,
)

MEN_ELO_PARAMS = {
    "initial_rating": 1977.6666265290937,
    "k_factor": 23.495826335613152,
    "home_advantage": 113.5749262336082,
    "season_carryover": 0.9087638652500951,
    "scale": 1718.0250410176775,
    "mov_alpha": 1.0940385366136334,
    "weight_regular": 0.5384629491188666,
    "weight_tourney": 0.5460196127895874,
}

WOMEN_ELO_PARAMS = {
    "initial_rating": 1213.0,
    "k_factor": 136.0,
    "home_advantage": 119.0,
    "season_carryover": 0.9840,
    "scale": 1888.27,
    "mov_alpha": 11.4612,
    "weight_regular": 1.4974,
    "weight_tourney": 0.5002,
}


@dataclass(frozen=True)
class FrozenMenContext:
    regular_season: pd.DataFrame
    regular_season_detailed: pd.DataFrame
    results: pd.DataFrame
    seeds: pd.DataFrame
    slots: pd.DataFrame
    conf_tourney: pd.DataFrame
    team_features: pd.DataFrame
    training: pd.DataFrame


@dataclass(frozen=True)
class FrozenWomenContext:
    regular_season: pd.DataFrame
    results: pd.DataFrame
    seeds: pd.DataFrame
    slots: pd.DataFrame
    elo_ratings: pd.DataFrame
    training: pd.DataFrame


def _load_men_reference_helpers() -> dict[str, Any]:
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    scripts_path = str(scripts_dir)
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)

    module = importlib.import_module("run_men_reference_margin")

    return {
        "apply_temperature": module._apply_temperature,
        "blend_probabilities": module._blend_probabilities,
        "drop_required_missing": module._drop_required_missing,
        "elo_probabilities": module._elo_probabilities,
        "estimate_residual_sigma": module._estimate_residual_sigma,
        "fit_reference_calibration": module._fit_reference_calibration,
        "margin_to_probability": module._margin_to_probability,
        "mirror_margin_training_rows": module._mirror_margin_training_rows,
        "prepare_reference_feature_frame": module._prepare_reference_feature_frame,
        "build_margin_pipeline": module.build_margin_pipeline,
    }


def build_seeded_submission_rows(seeds: pd.DataFrame, *, season: int) -> pd.DataFrame:
    """Build all seeded team-pair rows for a season in Kaggle submission orientation."""
    season_seeds = seeds.loc[seeds["Season"] == season, ["TeamID"]].copy()
    if season_seeds.empty:
        raise ValueError(f"No seed rows found for season {season}.")

    team_ids = sorted(int(team_id) for team_id in season_seeds["TeamID"].tolist())
    rows: list[dict[str, int | str]] = []
    for index, low_team in enumerate(team_ids):
        for high_team in team_ids[index + 1 :]:
            rows.append(
                {
                    "ID": f"{season}_{low_team}_{high_team}",
                    "Season": season,
                    "LowTeamID": low_team,
                    "HighTeamID": high_team,
                }
            )
    return pd.DataFrame(rows)


def load_frozen_men_context(data_dir: Path) -> FrozenMenContext:
    """Load reusable data and training features for the frozen men's model."""
    helpers = _load_men_reference_helpers()
    regular_season = pd.read_csv(data_dir / "MRegularSeasonCompactResults.csv")
    regular_season_detailed = pd.read_csv(data_dir / "MRegularSeasonDetailedResults.csv")
    results = pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv")
    seeds = pd.read_csv(data_dir / "MNCAATourneySeeds.csv")
    slots = pd.read_csv(data_dir / "MNCAATourneySlots.csv")
    conf_tourney = pd.read_csv(data_dir / "MConferenceTourneyGames.csv")
    season_floor = 2004

    regular_season = regular_season.loc[regular_season["Season"] >= season_floor].copy()
    regular_season_detailed = regular_season_detailed.loc[
        regular_season_detailed["Season"] >= season_floor
    ].copy()
    results = results.loc[results["Season"] >= season_floor].copy()
    seeds = seeds.loc[seeds["Season"] >= season_floor].copy()
    slots = slots.loc[slots["Season"] >= season_floor].copy()
    conf_tourney = conf_tourney.loc[conf_tourney["Season"] >= season_floor].copy()

    elo_ratings = compute_pre_tourney_elo_ratings(
        regular_season,
        tourney_results=results,
        day_cutoff=134,
        **MEN_ELO_PARAMS,
    )
    team_features = build_phase_ab_team_features(
        regular_season,
        regular_season_detailed,
        elo_ratings,
        conference_tourney_games=conf_tourney,
        day_cutoff=134,
        close_game_margin_threshold=1,
    )
    training = build_phase_ab_tourney_features(
        results,
        seeds,
        team_features,
        league="M",
        include_playins=True,
    )
    training = helpers["prepare_reference_feature_frame"](training)
    return FrozenMenContext(
        regular_season=regular_season,
        regular_season_detailed=regular_season_detailed,
        results=results,
        seeds=seeds,
        slots=slots,
        conf_tourney=conf_tourney,
        team_features=team_features,
        training=training,
    )


def load_frozen_women_context(data_dir: Path) -> FrozenWomenContext:
    """Load reusable data and training features for the frozen women's model."""
    regular_season = pd.read_csv(data_dir / "WRegularSeasonCompactResults.csv")
    results = pd.read_csv(data_dir / "WNCAATourneyCompactResults.csv")
    seeds = pd.read_csv(data_dir / "WNCAATourneySeeds.csv")
    slots = pd.read_csv(data_dir / "WNCAATourneySlots.csv")
    season_floor = 2010

    regular_season = regular_season.loc[regular_season["Season"] >= season_floor].copy()
    results = results.loc[results["Season"] >= season_floor].copy()
    seeds = seeds.loc[seeds["Season"] >= season_floor].copy()
    slots = slots.loc[slots["Season"] >= season_floor].copy()

    elo_ratings = compute_pre_tourney_elo_ratings(
        regular_season,
        tourney_results=results,
        day_cutoff=134,
        **WOMEN_ELO_PARAMS,
    )
    training = build_elo_seed_tourney_features(results, seeds, elo_ratings, league="W")
    return FrozenWomenContext(
        regular_season=regular_season,
        results=results,
        seeds=seeds,
        slots=slots,
        elo_ratings=elo_ratings,
        training=training,
    )


def predict_frozen_men(
    data_dir: Path,
    submission_rows: pd.DataFrame,
    season: int,
    *,
    context: FrozenMenContext | None = None,
) -> tuple[pd.DataFrame, BracketDiagnostics]:
    """Score arbitrary matchup rows with the frozen men's model."""
    helpers = _load_men_reference_helpers()
    context = context if context is not None else load_frozen_men_context(data_dir)
    results = context.results
    seeds = context.seeds
    slots = context.slots
    team_features = context.team_features
    training = context.training
    submission_frame = build_phase_ab_submission_features(
        submission_rows[["Season", "LowTeamID", "HighTeamID"]],
        seeds,
        team_features,
        league="M",
    )
    submission_frame = helpers["prepare_reference_feature_frame"](submission_frame)

    feature_cols = [
        "adj_qg_diff",
        "mov_per100_diff",
        "seed_diff",
        "ast_rate_diff",
        "close_win_pct_diff",
        "conf_tourney_win_pct_diff",
        "ftr_diff",
        "tov_rate_diff",
    ]
    calibration_feature_cols = [
        "adj_qg_diff",
        "mov_per100_diff",
        "seed_diff",
        "ast_rate_diff",
    ]

    train_frame = helpers["drop_required_missing"](training.loc[training["Season"] < season].copy())
    model = helpers["build_margin_pipeline"](feature_cols)
    mirrored_train = helpers["mirror_margin_training_rows"](train_frame, feature_cols)
    model.fit(mirrored_train[feature_cols], mirrored_train["margin"])
    sigma = helpers["estimate_residual_sigma"](model, mirrored_train, feature_cols)
    infer_margin = pd.Series(
        model.predict(submission_frame[feature_cols]),
        index=submission_frame.index,
    )
    submission_frame["p_raw"] = helpers["margin_to_probability"](infer_margin, sigma)
    calibration_state = helpers["fit_reference_calibration"](
        frame=training,
        full_feature_cols=feature_cols,
        calibration_feature_cols=calibration_feature_cols,
        target_season=season,
        seeds=seeds,
        team_features=team_features,
        elo_scale=MEN_ELO_PARAMS["scale"],
    )
    submission_frame["p_cal"] = helpers["apply_temperature"](
        submission_frame["p_raw"],
        calibration_state.temperature,
    )
    submission_frame["p_elo"] = helpers["elo_probabilities"](
        submission_frame,
        scale=MEN_ELO_PARAMS["scale"],
    )
    submission_frame["Pred"] = helpers["blend_probabilities"](
        submission_frame["p_cal"],
        submission_frame["p_elo"],
        calibration_state.alpha,
    )
    submission_frame["ID"] = submission_rows["ID"].to_numpy()

    bracket = compute_bracket_diagnostics(
        slots,
        seeds,
        submission_frame[["ID", "Pred"]],
        season=season,
        results=results,
        round_col="Round",
    )
    return submission_frame[["ID", "Pred", "Round", "round_group"]], bracket


def predict_frozen_women(
    data_dir: Path,
    submission_rows: pd.DataFrame,
    season: int,
    *,
    context: FrozenWomenContext | None = None,
) -> tuple[pd.DataFrame, BracketDiagnostics]:
    """Score arbitrary matchup rows with the frozen women's model."""
    context = context if context is not None else load_frozen_women_context(data_dir)
    results = context.results
    seeds = context.seeds
    slots = context.slots
    elo_ratings = context.elo_ratings
    training = context.training
    submission_frame = build_elo_seed_submission_features(
        submission_rows[["Season", "LowTeamID", "HighTeamID"]],
        seeds,
        elo_ratings,
        league="W",
    )

    train_frame = training.loc[training["Season"] < season].copy()
    model = build_logistic_pipeline(["seed_diff", "elo_diff"])
    model.fit(train_frame[["seed_diff", "elo_diff"]], train_frame["outcome"].astype(int))
    submission_frame["Pred"] = model.predict_proba(submission_frame[["seed_diff", "elo_diff"]])[
        :, 1
    ]
    submission_frame["ID"] = submission_rows["ID"].to_numpy()

    bracket = compute_bracket_diagnostics(
        slots,
        seeds,
        submission_frame[["ID", "Pred"]],
        season=season,
        results=results,
        round_col="Round",
    )
    return submission_frame[["ID", "Pred", "Round", "round_group"]], bracket
