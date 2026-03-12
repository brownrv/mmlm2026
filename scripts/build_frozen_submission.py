from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from run_men_reference_margin import (
    _apply_temperature,
    _blend_probabilities,
    _drop_required_missing,
    _elo_probabilities,
    _estimate_residual_sigma,
    _fit_reference_calibration,
    _margin_to_probability,
    _mirror_margin_training_rows,
    _prepare_reference_feature_frame,
    build_margin_pipeline,
)

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts
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
from mmlm2026.submission.frozen import (
    assign_submission_league,
    default_submission_output_path,
    parse_submission_ids,
)
from mmlm2026.submission.validation import validate_submission_file
from mmlm2026.submission.writer import write_submission

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the combined frozen-model submission.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026/SampleSubmissionStage2.csv"),
    )
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/submissions"))
    parser.add_argument("--tag", default="frozen")
    parser.add_argument("--save-artifacts", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    sample_submission = pd.read_csv(args.sample_submission)
    parsed = parse_submission_ids(sample_submission)
    season_values = sorted(int(value) for value in parsed["Season"].unique())
    season = args.season if args.season is not None else season_values[0]
    if season_values != [season]:
        raise ValueError(
            "Sample submission must contain exactly one season. "
            f"Found {season_values}, requested {season}."
        )

    men_teams = pd.read_csv(args.data_dir / "MTeams.csv")
    women_teams = pd.read_csv(args.data_dir / "WTeams.csv")
    parsed = assign_submission_league(
        parsed,
        men_team_ids=set(men_teams["TeamID"].astype(int)),
        women_team_ids=set(women_teams["TeamID"].astype(int)),
    )

    men_rows = parsed.loc[
        parsed["league"] == "M",
        ["ID", "Season", "LowTeamID", "HighTeamID"],
    ].copy()
    women_rows = parsed.loc[
        parsed["league"] == "W",
        ["ID", "Season", "LowTeamID", "HighTeamID"],
    ].copy()

    men_predictions, men_bracket = _predict_men(args.data_dir, men_rows, season)
    women_predictions, women_bracket = _predict_women(args.data_dir, women_rows, season)

    combined = (
        pd.concat(
            [men_predictions[["ID", "Pred"]], women_predictions[["ID", "Pred"]]],
            ignore_index=True,
        )
        .sort_values("ID")
        .reset_index(drop=True)
    )

    output_path = default_submission_output_path(
        season=season,
        output_dir=args.output_dir,
        tag=args.tag,
    )
    write_submission(combined, output_path)
    validate_submission_file(output_path, sample_submission_path=args.sample_submission)

    if args.save_artifacts:
        artifacts_dir = args.output_dir / f"{season}_{args.tag}_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        men_predictions.to_parquet(artifacts_dir / "men_predictions.parquet", index=False)
        women_predictions.to_parquet(artifacts_dir / "women_predictions.parquet", index=False)
        save_bracket_artifacts(men_bracket, output_dir=artifacts_dir / "men_bracket")
        save_bracket_artifacts(women_bracket, output_dir=artifacts_dir / "women_bracket")

    print(f"Wrote validated submission to {output_path}")
    print(f"Men rows: {len(men_predictions)} | Women rows: {len(women_predictions)}")
    return 0


def _predict_men(
    data_dir: Path,
    submission_rows: pd.DataFrame,
    season: int,
):
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
    training = _prepare_reference_feature_frame(training)
    submission_frame = build_phase_ab_submission_features(
        submission_rows[["Season", "LowTeamID", "HighTeamID"]],
        seeds,
        team_features,
        league="M",
    )
    submission_frame = _prepare_reference_feature_frame(submission_frame)

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

    train_frame = _drop_required_missing(training.loc[training["Season"] < season].copy())
    model = build_margin_pipeline(feature_cols)
    mirrored_train = _mirror_margin_training_rows(train_frame, feature_cols)
    model.fit(mirrored_train[feature_cols], mirrored_train["margin"])
    sigma = _estimate_residual_sigma(model, mirrored_train, feature_cols)
    infer_margin = pd.Series(
        model.predict(submission_frame[feature_cols]),
        index=submission_frame.index,
    )
    submission_frame["p_raw"] = _margin_to_probability(infer_margin, sigma)
    calibration_state = _fit_reference_calibration(
        frame=training,
        full_feature_cols=feature_cols,
        calibration_feature_cols=calibration_feature_cols,
        target_season=season,
        seeds=seeds,
        team_features=team_features,
        elo_scale=MEN_ELO_PARAMS["scale"],
    )
    submission_frame["p_cal"] = _apply_temperature(
        submission_frame["p_raw"],
        calibration_state.temperature,
    )
    submission_frame["p_elo"] = _elo_probabilities(
        submission_frame,
        scale=MEN_ELO_PARAMS["scale"],
    )
    submission_frame["Pred"] = _blend_probabilities(
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


def _predict_women(
    data_dir: Path,
    submission_rows: pd.DataFrame,
    season: int,
):
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


if __name__ == "__main__":
    raise SystemExit(main())
