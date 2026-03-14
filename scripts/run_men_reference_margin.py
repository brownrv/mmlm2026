from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts
from mmlm2026.evaluation.validation import (
    ValidationSummary,
    build_calibration_table,
    save_validation_artifacts,
)
from mmlm2026.features.elo import (
    compute_elo_momentum_features,
    compute_pre_tourney_elo_ratings,
    compute_tournament_only_elo_ratings,
)
from mmlm2026.features.espn import (
    load_espn_men_four_factor_strength_features,
    load_espn_men_rotation_stability_features,
)
from mmlm2026.features.primary import (
    build_conference_percentile_features,
    build_late5_form_split_features,
    build_market_implied_strength_features,
    build_phase_ab_matchup_features,
    build_phase_ab_team_features,
    build_phase_ab_tourney_features,
    build_program_pedigree_features,
    build_season_momentum_features,
    build_site_performance_features,
    build_team_season_summary,
    build_win_quality_bin_features,
)
from mmlm2026.utils.mlflow_tracking import start_tracked_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the men reference-style margin regression challenger."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument("--holdout-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/reference_runs"))
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--run-name", default="val-05-men-reference-margin")
    parser.add_argument("--day-cutoff", type=int, default=134)
    parser.add_argument("--season-floor", type=int, default=2004)
    parser.add_argument(
        "--market-root",
        type=Path,
        default=Path("data/processed/betexplorer"),
    )
    parser.add_argument(
        "--espn-root",
        type=Path,
        default=Path("data/processed/espn/mens-college-basketball"),
    )
    parser.add_argument(
        "--team-spellings-path",
        type=Path,
        default=Path("data/TeamSpellings.csv"),
    )
    parser.add_argument(
        "--include-ridge-strength",
        action="store_true",
        help="Include ridge_strength_diff from a regularized margin rating as an extra feature.",
    )
    parser.add_argument(
        "--include-espn-four-factor",
        action="store_true",
        help="Include ESPN-derived four-factor strength differential as an extra feature.",
    )
    parser.add_argument(
        "--include-espn-components",
        action="store_true",
        help="Include ESPN-derived four-factor component differentials as extra features.",
    )
    parser.add_argument(
        "--include-espn-rotation",
        action="store_true",
        help="Include ESPN-derived rotation-stability differential as an extra feature.",
    )
    parser.add_argument(
        "--include-tourney-elo",
        action="store_true",
        help="Include tournament-only Elo differential as an extra feature.",
    )
    parser.add_argument(
        "--include-elo-momentum",
        action="store_true",
        help="Include Elo momentum differential from DayNum-115 to pre-tournament Elo.",
    )
    parser.add_argument(
        "--include-pythag",
        action="store_true",
        help="Include Pythagorean expectancy differential as an extra feature.",
    )
    parser.add_argument(
        "--include-seed-elo-gap",
        action="store_true",
        help="Include seed-Elo gap differential as an extra feature.",
    )
    parser.add_argument(
        "--include-season-momentum",
        action="store_true",
        help=(
            "Include second-half minus first-half average margin "
            "differential as an extra feature."
        ),
    )
    parser.add_argument(
        "--include-market-strength",
        action="store_true",
        help="Include regular-season BetExplorer market-implied strength differential.",
    )
    parser.add_argument(
        "--include-late-bundle",
        action="store_true",
        help="Include the bundled late challenger feature set (24/27/28/29/31).",
    )
    parser.add_argument(
        "--use-decay-weighting",
        action="store_true",
        help="Apply exponential season-recency weights during training.",
    )
    parser.add_argument(
        "--decay-base",
        type=float,
        default=0.9,
        help="Per-season exponential decay base for training weights.",
    )
    parser.add_argument("--elo-initial-rating", type=float, default=1618.0)
    parser.add_argument("--elo-k-factor", type=float, default=76.0)
    parser.add_argument("--elo-home-advantage", type=float, default=43.0)
    parser.add_argument("--elo-season-carryover", type=float, default=0.9780)
    parser.add_argument("--elo-scale", type=float, default=1835.34)
    parser.add_argument("--elo-mov-alpha", type=float, default=6.5450)
    parser.add_argument("--elo-weight-regular", type=float, default=1.4674)
    parser.add_argument("--elo-weight-tourney", type=float, default=0.8204)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    data_dir = args.data_dir
    regular_season = pd.read_csv(data_dir / "MRegularSeasonCompactResults.csv")
    regular_season_detailed = pd.read_csv(data_dir / "MRegularSeasonDetailedResults.csv")
    results = pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv")
    seeds = pd.read_csv(data_dir / "MNCAATourneySeeds.csv")
    slots = pd.read_csv(data_dir / "MNCAATourneySlots.csv")
    conf_tourney = pd.read_csv(data_dir / "MConferenceTourneyGames.csv")
    team_conferences = pd.read_csv(data_dir / "MTeamConferences.csv")

    regular_season = regular_season.loc[regular_season["Season"] >= args.season_floor].copy()
    regular_season_detailed = regular_season_detailed.loc[
        regular_season_detailed["Season"] >= args.season_floor
    ].copy()
    results = results.loc[results["Season"] >= args.season_floor].copy()
    seeds = seeds.loc[seeds["Season"] >= args.season_floor].copy()
    slots = slots.loc[slots["Season"] >= args.season_floor].copy()
    conf_tourney = conf_tourney.loc[conf_tourney["Season"] >= args.season_floor].copy()
    team_conferences = team_conferences.loc[
        team_conferences["Season"] >= args.season_floor
    ].copy()

    elo_ratings = compute_pre_tourney_elo_ratings(
        regular_season,
        tourney_results=results,
        day_cutoff=args.day_cutoff,
        initial_rating=args.elo_initial_rating,
        k_factor=args.elo_k_factor,
        home_advantage=args.elo_home_advantage,
        season_carryover=args.elo_season_carryover,
        scale=args.elo_scale,
        mov_alpha=args.elo_mov_alpha,
        weight_regular=args.elo_weight_regular,
        weight_tourney=args.elo_weight_tourney,
    )
    team_features = build_phase_ab_team_features(
        regular_season,
        regular_season_detailed,
        elo_ratings,
        conference_tourney_games=conf_tourney,
        day_cutoff=args.day_cutoff,
        close_game_margin_threshold=1,
    )
    if args.include_tourney_elo:
        tourney_elo = compute_tournament_only_elo_ratings(results, seeds=seeds).rename(
            columns={"elo": "tourney_elo"}
        )
        team_features = team_features.merge(
            tourney_elo,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
    if args.include_elo_momentum:
        elo_momentum = compute_elo_momentum_features(
            regular_season,
            mid_day_cutoff=115,
            end_day_cutoff=args.day_cutoff,
            initial_rating=args.elo_initial_rating,
            k_factor=args.elo_k_factor,
            home_advantage=args.elo_home_advantage,
            season_carryover=args.elo_season_carryover,
            scale=args.elo_scale,
            mov_alpha=args.elo_mov_alpha,
            weight_regular=args.elo_weight_regular,
        )[["Season", "TeamID", "elo_momentum"]]
        team_features = team_features.merge(
            elo_momentum,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
    if args.include_season_momentum:
        season_momentum = build_season_momentum_features(
            regular_season,
            day_cutoff=args.day_cutoff,
        )
        team_features = team_features.merge(
            season_momentum,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
    if args.include_market_strength:
        market_regular = pd.read_parquet(args.market_root / "ncaam_regular.parquet")
        market_regular = market_regular.loc[market_regular["Season"] >= args.season_floor].copy()
        market_strength = build_market_implied_strength_features(
            market_regular,
            day_cutoff=args.day_cutoff,
        )
        team_features = team_features.merge(
            market_strength,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
    if args.include_late_bundle:
        late5 = build_late5_form_split_features(
            regular_season_detailed,
            day_floor=115,
            day_cutoff=args.day_cutoff,
        )
        site_profiles = build_site_performance_features(
            regular_season,
            day_cutoff=args.day_cutoff,
        )
        win_quality = build_win_quality_bin_features(
            regular_season,
            day_cutoff=args.day_cutoff,
        )
        summary_strength = build_team_season_summary(regular_season, day_cutoff=args.day_cutoff)[
            ["Season", "TeamID", "avg_margin"]
        ]
        conf_rank = build_conference_percentile_features(
            summary_strength,
            team_conferences,
            strength_col="avg_margin",
        )
        pedigree = build_program_pedigree_features(seeds, results)
        team_features = (
            team_features.merge(late5, on=["Season", "TeamID"], how="left", validate="one_to_one")
            .merge(
                site_profiles,
                on=["Season", "TeamID"],
                how="left",
                validate="one_to_one",
            )
            .merge(win_quality, on=["Season", "TeamID"], how="left", validate="one_to_one")
            .merge(conf_rank, on=["Season", "TeamID"], how="left", validate="one_to_one")
            .merge(pedigree, on=["Season", "TeamID"], how="left", validate="one_to_one")
        )
    if args.include_espn_four_factor:
        seasons = sorted(
            season
            for season in team_features["Season"].drop_duplicates().astype(int).tolist()
            if season <= max(args.holdout_seasons)
        )
        espn_features = load_espn_men_four_factor_strength_features(
            espn_root=args.espn_root,
            seasons=seasons,
            regular_season_results=regular_season_detailed,
            team_spellings_path=args.team_spellings_path,
        )
        team_features = team_features.merge(
            espn_features,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
    if args.include_espn_components and not args.include_espn_four_factor:
        seasons = sorted(
            season
            for season in team_features["Season"].drop_duplicates().astype(int).tolist()
            if season <= max(args.holdout_seasons)
        )
        espn_features = load_espn_men_four_factor_strength_features(
            espn_root=args.espn_root,
            seasons=seasons,
            regular_season_results=regular_season_detailed,
            team_spellings_path=args.team_spellings_path,
        )
        team_features = team_features.merge(
            espn_features,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
    if args.include_espn_rotation:
        seasons = sorted(
            season
            for season in team_features["Season"].drop_duplicates().astype(int).tolist()
            if season <= max(args.holdout_seasons)
        )
        espn_rotation = load_espn_men_rotation_stability_features(
            espn_root=args.espn_root,
            seasons=seasons,
            regular_season_results=regular_season_detailed,
            team_spellings_path=args.team_spellings_path,
        )
        team_features = team_features.merge(
            espn_rotation,
            on=["Season", "TeamID"],
            how="left",
            validate="one_to_one",
        )
    feature_table = build_phase_ab_tourney_features(
        results,
        seeds,
        team_features,
        league="M",
        include_playins=True,
    )
    feature_table = _prepare_reference_feature_frame(feature_table)
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
    if args.include_ridge_strength:
        feature_cols.append("ridge_strength_diff")
    if args.include_espn_four_factor:
        feature_cols.append("espn_four_factor_strength_diff")
    if args.include_espn_components:
        feature_cols.extend(
            [
                "espn_efg_diff",
                "espn_tov_rate_diff",
                "espn_orb_pct_diff",
                "espn_ftr_diff",
                "espn_opp_efg_diff",
                "espn_opp_tov_rate_diff",
                "espn_opp_orb_pct_diff",
                "espn_opp_ftr_diff",
            ]
        )
    if args.include_espn_rotation:
        feature_cols.append("espn_rotation_stability_diff")
    if args.include_tourney_elo:
        feature_cols.append("tourney_elo_diff")
    if args.include_elo_momentum:
        feature_cols.append("elo_momentum_diff")
    if args.include_pythag:
        feature_cols.append("pythag_diff")
    if args.include_seed_elo_gap:
        feature_cols.append("seed_elo_gap_diff")
    if args.include_season_momentum:
        feature_cols.append("season_momentum_diff")
    if args.include_market_strength:
        feature_cols.append("market_implied_strength_diff")
    if args.include_late_bundle:
        feature_cols.extend(
            [
                "late5_off_diff",
                "late5_def_diff",
                "road_win_pct_diff",
                "neutral_net_eff_diff",
                "close_win_pct_5_diff",
                "blowout_win_pct_diff",
                "conf_pct_rank_diff",
                "pedigree_score_diff",
            ]
        )
    calibration_feature_cols = [
        "adj_qg_diff",
        "mov_per100_diff",
        "seed_diff",
        "ast_rate_diff",
    ]
    if args.include_ridge_strength:
        calibration_feature_cols.append("ridge_strength_diff")
    if args.include_espn_four_factor:
        calibration_feature_cols.append("espn_four_factor_strength_diff")
    if args.include_espn_components:
        calibration_feature_cols.extend(
            [
                "espn_efg_diff",
                "espn_tov_rate_diff",
                "espn_orb_pct_diff",
                "espn_ftr_diff",
                "espn_opp_efg_diff",
                "espn_opp_tov_rate_diff",
                "espn_opp_orb_pct_diff",
                "espn_opp_ftr_diff",
            ]
        )
    if args.include_espn_rotation:
        calibration_feature_cols.append("espn_rotation_stability_diff")
    if args.include_tourney_elo:
        calibration_feature_cols.append("tourney_elo_diff")
    if args.include_elo_momentum:
        calibration_feature_cols.append("elo_momentum_diff")
    if args.include_pythag:
        calibration_feature_cols.append("pythag_diff")
    if args.include_seed_elo_gap:
        calibration_feature_cols.append("seed_elo_gap_diff")
    if args.include_season_momentum:
        calibration_feature_cols.append("season_momentum_diff")
    if args.include_market_strength:
        calibration_feature_cols.append("market_implied_strength_diff")
    if args.include_late_bundle:
        calibration_feature_cols.extend(
            [
                "late5_off_diff",
                "late5_def_diff",
                "road_win_pct_diff",
                "neutral_net_eff_diff",
                "close_win_pct_5_diff",
                "blowout_win_pct_diff",
                "conf_pct_rank_diff",
                "pedigree_score_diff",
            ]
        )

    output_dir = args.output_dir / "m_reference_margin"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_table.to_parquet(output_dir / "men_reference_features.parquet", index=False)
        team_features.to_parquet(output_dir / "phase_ab_team_features.parquet", index=False)

    summary = _validate_margin_holdouts(
        feature_table,
        feature_cols=feature_cols,
        holdout_seasons=args.holdout_seasons,
        calibration_feature_cols=calibration_feature_cols,
        seeds=seeds,
        team_features=team_features,
        elo_scale=args.elo_scale,
        use_decay_weighting=args.use_decay_weighting,
        decay_base=args.decay_base,
    )
    validation_artifacts = save_validation_artifacts(summary, output_dir=output_dir / "validation")

    latest_holdout = max(args.holdout_seasons)
    train_frame = _drop_required_missing(
        feature_table.loc[feature_table["Season"] < latest_holdout].copy()
    )
    infer_frame = build_phase_ab_matchup_features(
        seeds,
        team_features,
        season=latest_holdout,
        league="M",
    )
    infer_frame = _prepare_reference_feature_frame(infer_frame)
    model = build_margin_pipeline(feature_cols)
    mirrored_train = _mirror_margin_training_rows(train_frame, feature_cols)
    model.fit(mirrored_train[feature_cols], mirrored_train["margin"])
    sigma = _estimate_residual_sigma(model, mirrored_train, feature_cols)
    infer_margin = pd.Series(model.predict(infer_frame[feature_cols]), index=infer_frame.index)
    infer_frame["p_raw"] = _margin_to_probability(infer_margin, sigma)
    calibration_state = _fit_reference_calibration(
        frame=feature_table,
        full_feature_cols=feature_cols,
        calibration_feature_cols=calibration_feature_cols,
        target_season=latest_holdout,
        seeds=seeds,
        team_features=team_features,
        elo_scale=args.elo_scale,
    )
    infer_frame["p_cal"] = _apply_temperature(infer_frame["p_raw"], calibration_state.temperature)
    infer_frame["p_elo"] = _elo_probabilities(infer_frame, scale=args.elo_scale)
    infer_frame["Pred"] = _blend_probabilities(
        infer_frame["p_cal"],
        infer_frame["p_elo"],
        calibration_state.alpha,
    )
    infer_frame["ID"] = [
        f"{latest_holdout}_{low}_{high}"
        for low, high in zip(infer_frame["LowTeamID"], infer_frame["HighTeamID"], strict=False)
    ]

    bracket = compute_bracket_diagnostics(
        slots,
        seeds,
        infer_frame[["ID", "Pred"]],
        season=latest_holdout,
        results=results,
        round_col="Round",
    )
    bracket_artifacts = save_bracket_artifacts(bracket, output_dir=output_dir / "bracket")

    if args.log_mlflow:
        _log_mlflow_run(
            args,
            summary,
            validation_artifacts,
            bracket_artifacts,
            feature_cols,
            calibration_state.temperature,
            calibration_state.alpha,
        )

    print(summary.per_season_metrics.to_string(index=False))
    print(
        "\nOverall metrics:",
        f"flat_brier={summary.overall_flat_brier:.6f}",
        f"log_loss={summary.overall_log_loss:.6f}",
    )
    print(f"Artifacts written under {output_dir}")
    return 0


def build_margin_pipeline(feature_cols: list[str]) -> Pipeline:
    optional_feature_cols = [
        col for col in feature_cols if col not in {"adj_qg_diff", "mov_per100_diff", "seed_diff"}
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "required_numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]),
                ["adj_qg_diff", "mov_per100_diff", "seed_diff"],
            ),
            (
                "optional_numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]),
                optional_feature_cols,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    max_depth=3,
                    learning_rate=0.05,
                    min_samples_leaf=20,
                    random_state=42,
                ),
            ),
        ]
    )


def _validate_margin_holdouts(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    holdout_seasons: list[int],
    calibration_feature_cols: list[str],
    seeds: pd.DataFrame,
    team_features: pd.DataFrame,
    elo_scale: float,
    use_decay_weighting: bool = False,
    decay_base: float = 0.9,
    train_min_games: int = 50,
) -> ValidationSummary:
    work = frame.loc[frame["outcome"].notna() & frame["margin"].notna()].copy()
    metrics_rows: list[dict[str, float | int | None]] = []
    prediction_frames: list[pd.DataFrame] = []

    for holdout_season in holdout_seasons:
        train = work.loc[work["Season"] < holdout_season].copy()
        valid = work.loc[(work["Season"] == holdout_season) & (work["Round"] > 0)].copy()
        if len(train) < train_min_games:
            raise ValueError(
                f"Training set for season {holdout_season} has {len(train)} games; "
                f"minimum required is {train_min_games}."
            )

        model = build_margin_pipeline(feature_cols)
        mirrored_train = _mirror_margin_training_rows(train, feature_cols)
        fit_kwargs: dict[str, pd.Series] = {}
        if use_decay_weighting:
            fit_kwargs["model__sample_weight"] = _season_decay_weights(
                mirrored_train["Season"],
                decay_base=decay_base,
            )
        model.fit(mirrored_train[feature_cols], mirrored_train["margin"], **fit_kwargs)
        sigma = _estimate_residual_sigma(model, mirrored_train, feature_cols)
        pred_margin = pd.Series(model.predict(valid[feature_cols]), index=valid.index)
        pred_prob_raw = _margin_to_probability(pred_margin, sigma)
        calibration_state = _fit_reference_calibration(
            frame=frame,
            full_feature_cols=feature_cols,
            calibration_feature_cols=calibration_feature_cols,
            target_season=holdout_season,
            seeds=seeds,
            team_features=team_features,
            elo_scale=elo_scale,
            use_decay_weighting=use_decay_weighting,
            decay_base=decay_base,
        )
        pred_prob = _blend_probabilities(
            _apply_temperature(pred_prob_raw, calibration_state.temperature),
            _elo_probabilities(valid, scale=elo_scale),
            calibration_state.alpha,
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "Season": valid["Season"].to_numpy(),
                    "LowTeamID": valid["LowTeamID"].to_numpy(),
                    "HighTeamID": valid["HighTeamID"].to_numpy(),
                    "outcome": valid["outcome"].astype(int).to_numpy(),
                    "pred": pred_prob.to_numpy(),
                    "round_group": valid["round_group"].to_numpy(),
                    "Round": valid["Round"].to_numpy(),
                }
            )
        )
        season_pred = prediction_frames[-1]
        metrics_rows.append(
            {
                "Season": holdout_season,
                "train_tourney_games": len(train),
                "valid_tourney_games": len(valid),
                "flat_brier": float(brier_score_loss(season_pred["outcome"], season_pred["pred"])),
                "log_loss": float(log_loss(season_pred["outcome"], season_pred["pred"])),
                "r1_brier": _group_brier(season_pred, "R1"),
                "r2plus_brier": _group_brier(season_pred, "R2+"),
            }
        )

    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    calibration_table = build_calibration_table(
        oof_predictions["outcome"],
        oof_predictions["pred"],
        n_bins=10,
    )
    return ValidationSummary(
        per_season_metrics=pd.DataFrame(metrics_rows).sort_values("Season").reset_index(drop=True),
        calibration_table=calibration_table,
        oof_predictions=oof_predictions,
    )


def _mirror_margin_training_rows(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    original = frame[["Season", *feature_cols, "margin"]].copy()
    mirrored = original.copy()
    for col in feature_cols:
        mirrored[col] = -mirrored[col]
    mirrored["margin"] = -mirrored["margin"]
    return pd.concat([original, mirrored], ignore_index=True)


def _season_decay_weights(seasons: pd.Series, *, decay_base: float) -> pd.Series:
    latest = int(seasons.max())
    weights = seasons.astype(int).map(
        lambda season: float(decay_base ** (latest - int(season)))
    )
    return weights.astype(float)


def _estimate_residual_sigma(
    model: Any,
    train_frame: pd.DataFrame,
    feature_cols: list[str],
) -> float:
    pred_margin = pd.Series(model.predict(train_frame[feature_cols]), index=train_frame.index)
    residual = train_frame["margin"].astype(float) - pred_margin
    return max(float(residual.std(ddof=0)), 0.5)


def _margin_to_probability(margin: pd.Series, sigma: float) -> pd.Series:
    normal = NormalDist()
    return pd.Series(
        [normal.cdf(float(value) / sigma) for value in margin],
        index=margin.index,
        dtype=float,
    ).clip(lower=1e-10, upper=1.0 - 1e-10)


class CalibrationState:
    def __init__(self, temperature: float, alpha: float) -> None:
        self.temperature = temperature
        self.alpha = alpha


def _prepare_reference_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if {"low_seed", "high_seed"}.issubset(prepared.columns):
        prepared["seed_diff"] = prepared["low_seed"] - prepared["high_seed"]
    return prepared


def _drop_required_missing(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["adj_qg_diff", "mov_per100_diff", "seed_diff"]
    return frame.dropna(subset=required).reset_index(drop=True)


def _logit(probabilities: pd.Series) -> pd.Series:
    clipped = probabilities.astype(float).clip(lower=1e-10, upper=1.0 - 1e-10)
    return np.log(clipped / (1.0 - clipped))


def _apply_temperature(probabilities: pd.Series, temperature: float) -> pd.Series:
    logits = _logit(probabilities) / temperature
    calibrated = 1.0 / (1.0 + np.exp(-logits))
    return pd.Series(calibrated, index=probabilities.index, dtype=float).clip(
        lower=1e-10,
        upper=1.0 - 1e-10,
    )


def _blend_probabilities(
    p_cal: pd.Series,
    p_elo: pd.Series,
    alpha: float,
) -> pd.Series:
    blended = alpha * p_elo.astype(float) + (1.0 - alpha) * p_cal.astype(float)
    return blended.clip(lower=1e-10, upper=1.0 - 1e-10)


def _elo_probabilities(frame: pd.DataFrame, *, scale: float) -> pd.Series:
    elo_diff = frame["low_elo"].astype(float) - frame["high_elo"].astype(float)
    probabilities = 1.0 / (1.0 + 10.0 ** ((-elo_diff) / scale))
    return pd.Series(probabilities, index=frame.index, dtype=float).clip(
        lower=1e-10,
        upper=1.0 - 1e-10,
    )


def _fit_reference_calibration(
    *,
    frame: pd.DataFrame,
    full_feature_cols: list[str],
    calibration_feature_cols: list[str],
    target_season: int,
    seeds: pd.DataFrame,
    team_features: pd.DataFrame,
    elo_scale: float,
    use_decay_weighting: bool = False,
    decay_base: float = 0.9,
    cal_seasons: int = 8,
    alpha_fallback: float = 0.45,
) -> CalibrationState:
    prior_seasons = sorted(
        int(season) for season in frame["Season"].unique() if int(season) < target_season
    )
    calibration_seasons = prior_seasons[-cal_seasons:]
    if not calibration_seasons:
        return CalibrationState(temperature=1.0, alpha=alpha_fallback)

    t_rows: list[dict[str, float]] = []
    pooled_probabilities: list[float] = []
    pooled_outcomes: list[int] = []
    alpha_values: list[float] = []
    alpha_grid = np.linspace(0.0, 1.0, 101)

    for season in calibration_seasons:
        train = frame.loc[frame["Season"] < season].copy()
        valid = frame.loc[(frame["Season"] == season) & (frame["Round"] > 0)].copy()
        train = _drop_required_missing(train)
        valid = _drop_required_missing(valid)
        if train.empty or valid.empty:
            continue

        v4_model = build_margin_pipeline(calibration_feature_cols)
        mirrored_v4 = _mirror_margin_training_rows(train, calibration_feature_cols)
        fit_kwargs: dict[str, pd.Series] = {}
        if use_decay_weighting:
            fit_kwargs["model__sample_weight"] = _season_decay_weights(
                mirrored_v4["Season"],
                decay_base=decay_base,
            )
        v4_model.fit(mirrored_v4[calibration_feature_cols], mirrored_v4["margin"], **fit_kwargs)
        sigma_v4 = _estimate_residual_sigma(v4_model, mirrored_v4, calibration_feature_cols)
        valid_margin = pd.Series(
            v4_model.predict(valid[calibration_feature_cols]),
            index=valid.index,
        )
        p_raw = _margin_to_probability(valid_margin, sigma_v4)

        optimal_t = _optimize_temperature(p_raw, valid["outcome"].astype(int))
        eff_sigma = _field_eff_sigma(season=season, seeds=seeds, team_features=team_features)
        t_rows.append({"eff_sigma": eff_sigma, "temperature": optimal_t})
        pooled_probabilities.extend(p_raw.tolist())
        pooled_outcomes.extend(valid["outcome"].astype(int).tolist())

        p_elo = _elo_probabilities(valid, scale=elo_scale)
        best_alpha = min(
            alpha_grid,
            key=lambda alpha: float(
                brier_score_loss(
                    valid["outcome"].astype(int),
                    _blend_probabilities(p_raw, p_elo, float(alpha)),
                )
            ),
        )
        alpha_values.append(float(best_alpha))

    if pooled_probabilities and pooled_outcomes:
        pooled_t = _optimize_temperature(
            pd.Series(pooled_probabilities, dtype=float),
            pd.Series(pooled_outcomes, dtype=int),
        )
    else:
        pooled_t = 1.0

    if len(t_rows) >= 2:
        t_frame = pd.DataFrame(t_rows)
        coeffs = np.polyfit(t_frame["eff_sigma"], t_frame["temperature"], deg=1)
        target_eff_sigma = _field_eff_sigma(
            season=target_season,
            seeds=seeds,
            team_features=team_features,
        )
        dynamic_t = float(coeffs[0] * target_eff_sigma + coeffs[1])
        correlation = float(abs(t_frame["eff_sigma"].corr(t_frame["temperature"])))
        weight = min(max(correlation, 0.0), 1.0) if not np.isnan(correlation) else 0.0
        final_t = weight * dynamic_t + (1.0 - weight) * pooled_t
    else:
        final_t = pooled_t

    final_t = float(np.clip(final_t, 0.5, 3.0))
    final_alpha = float(np.mean(alpha_values)) if len(alpha_values) >= 5 else alpha_fallback
    return CalibrationState(temperature=final_t, alpha=final_alpha)


def _optimize_temperature(probabilities: pd.Series, outcomes: pd.Series) -> float:
    grid = np.linspace(0.1, 5.0, 200)
    return float(
        min(
            grid,
            key=lambda temperature: float(
                log_loss(
                    outcomes.astype(int),
                    _apply_temperature(probabilities, float(temperature)),
                )
            ),
        )
    )


def _field_eff_sigma(
    *,
    season: int,
    seeds: pd.DataFrame,
    team_features: pd.DataFrame,
) -> float:
    field = seeds.loc[seeds["Season"] == season, ["TeamID"]].drop_duplicates()
    field_features = field.merge(
        team_features.loc[team_features["Season"] == season, ["TeamID", "adj_net_eff"]],
        on="TeamID",
        how="left",
        validate="one_to_one",
    )
    return float(field_features["adj_net_eff"].astype(float).std(ddof=0))


def _group_brier(frame: pd.DataFrame, round_group: str) -> float | None:
    mask = frame["round_group"] == round_group
    if not mask.any():
        return None
    return float(brier_score_loss(frame.loc[mask, "outcome"], frame.loc[mask, "pred"]))


def _log_mlflow_run(
    args: argparse.Namespace,
    summary: ValidationSummary,
    validation_artifacts,
    bracket_artifacts,
    feature_cols: list[str],
    temperature: float,
    alpha: float,
) -> None:
    tags = {
        "hypothesis": (
            "Men situational features plus margin regression may close the "
            "benchmark gap to the reference challenger"
        ),
        "model_family": "hist_gradient_boosting_regressor",
        "league": "men",
        "season_window": f"pre-{min(args.holdout_seasons)}",
        "depends_on": (
            "feature:feat_12_iter_adj_qg_v1,feature:men_situational_v1,"
            "feature:elo_tuned_carryover_men_v1"
            + (",feature:late_rate_01_ridge_strength_v1" if args.include_ridge_strength else "")
            + (",feature:late_feat_18_espn_four_factor_v1" if args.include_espn_four_factor else "")
            + (",feature:late_ext_03_espn_components_v1" if args.include_espn_components else "")
            + (",feature:late_feat_19_espn_rotation_v1" if args.include_espn_rotation else "")
            + (",feature:late_feat_21_tourney_elo_v1" if args.include_tourney_elo else "")
            + (",feature:late_feat_22_elo_momentum_v1" if args.include_elo_momentum else "")
            + (",feature:late_feat_26_pythag_expectancy_v1" if args.include_pythag else "")
            + (",feature:late_feat_23_seed_elo_gap_v1" if args.include_seed_elo_gap else "")
            + (",feature:late_feat_25_season_momentum_v1" if args.include_season_momentum else "")
            + (
                ",feature:late_feat_16_market_implied_strength_v1"
                if args.include_market_strength
                else ""
            )
            + (",feature:late_feat_bundle_24_27_28_29_31_v1" if args.include_late_bundle else "")
            + (",arch:late_arch_dw_01_v1" if args.use_decay_weighting else "")
        ),
        "retest_if": "men situational features or margin-to-probability conversion change",
        "leakage_audit": "passed",
    }
    with start_tracked_run(args.run_name, tags=tags):
        mlflow.log_params(
            {
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "features": ",".join(feature_cols),
                "season_floor": args.season_floor,
                "elo_day_cutoff": args.day_cutoff,
                "model": "HistGradientBoostingRegressor",
                "margin_to_probability": "gaussian_cdf",
                "include_ridge_strength": str(args.include_ridge_strength).lower(),
                "include_espn_four_factor": str(args.include_espn_four_factor).lower(),
                "include_espn_components": str(args.include_espn_components).lower(),
                "include_espn_rotation": str(args.include_espn_rotation).lower(),
                "include_tourney_elo": str(args.include_tourney_elo).lower(),
                "include_elo_momentum": str(args.include_elo_momentum).lower(),
                "include_pythag": str(args.include_pythag).lower(),
                "include_seed_elo_gap": str(args.include_seed_elo_gap).lower(),
                "include_season_momentum": str(args.include_season_momentum).lower(),
                "include_market_strength": str(args.include_market_strength).lower(),
                "include_late_bundle": str(args.include_late_bundle).lower(),
                "use_decay_weighting": str(args.use_decay_weighting).lower(),
                "decay_base": args.decay_base,
                "temperature": temperature,
                "alpha": alpha,
            }
        )
        mlflow.log_metrics(
            {
                "flat_brier": summary.overall_flat_brier,
                "log_loss": summary.overall_log_loss,
            }
        )
        r1_values = summary.per_season_metrics["r1_brier"].dropna()
        if not r1_values.empty:
            mlflow.log_metric("r1_brier_mean", float(r1_values.mean()))
        r2_values = summary.per_season_metrics["r2plus_brier"].dropna()
        if not r2_values.empty:
            mlflow.log_metric("r2plus_brier_mean", float(r2_values.mean()))

        for artifact_path in [
            validation_artifacts.per_season_metrics_path,
            validation_artifacts.calibration_table_path,
            validation_artifacts.oof_predictions_path,
            validation_artifacts.reliability_diagram_path,
            bracket_artifacts.play_prob_path,
            bracket_artifacts.slot_team_prob_path,
        ]:
            mlflow.log_artifact(str(artifact_path))
        if bracket_artifacts.bucket_summary_path is not None:
            mlflow.log_artifact(str(bracket_artifacts.bucket_summary_path))
        if bracket_artifacts.round_summary_path is not None:
            mlflow.log_artifact(str(bracket_artifacts.round_summary_path))


if __name__ == "__main__":
    raise SystemExit(main())
