from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss  # type: ignore[import-untyped]

from mmlm2026.evaluation.bracket import compute_bracket_diagnostics, save_bracket_artifacts
from mmlm2026.evaluation.men_round_group import route_group_predictions
from mmlm2026.evaluation.validation import (
    ValidationSummary,
    build_calibration_table,
    build_logistic_pipeline,
    save_validation_artifacts,
)
from mmlm2026.features.elo import (
    attach_secondary_elo_features,
    build_elo_seed_submission_features,
    build_elo_seed_tourney_features,
    compute_elo_momentum_features,
    compute_pre_tourney_elo_ratings,
    compute_tournament_only_elo_ratings,
)
from mmlm2026.features.espn import load_espn_women_four_factor_strength_features
from mmlm2026.features.phase_b import build_glm_quality_features
from mmlm2026.features.primary import (
    build_conference_percentile_features,
    build_late5_form_split_features,
    build_opponent_raw_boxscore_features,
    build_program_pedigree_features,
    build_season_momentum_features,
    build_site_performance_features,
    build_team_season_summary,
    build_win_quality_bin_features,
)
from mmlm2026.submission.frozen_models import (
    WOMEN_ELO_PARAMS,
    build_seeded_submission_rows,
    load_frozen_women_context,
)
from mmlm2026.utils.mlflow_tracking import start_tracked_run

ROUND_GROUPS = ("R1", "R2+")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the routed women R1-vs-R2+ late challenger.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/march-machine-learning-mania-2026"),
    )
    parser.add_argument("--holdout-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/challenger_runs"))
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--run-name", default="late-arch-rg-08-women-routed-round-group")
    parser.add_argument(
        "--espn-root",
        type=Path,
        default=Path("data/processed/espn/womens-college-basketball"),
    )
    parser.add_argument(
        "--team-spellings-path",
        type=Path,
        default=Path("data/TeamSpellings.csv"),
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
            "Include second-half minus first-half average margin differential as an extra feature."
        ),
    )
    parser.add_argument(
        "--include-espn-four-factor",
        action="store_true",
        help="Include ESPN-derived women four-factor strength differential.",
    )
    parser.add_argument(
        "--include-espn-components",
        action="store_true",
        help="Include ESPN-derived women four-factor component differentials.",
    )
    parser.add_argument(
        "--include-late5-form",
        action="store_true",
        help="Include late-5 offense/defense split differentials as extra features.",
    )
    parser.add_argument(
        "--include-conference-rank",
        action="store_true",
        help="Include conference percentile rank differential as an extra feature.",
    )
    parser.add_argument(
        "--include-pace",
        action="store_true",
        help="Include team pace differential as an extra feature.",
    )
    parser.add_argument(
        "--include-opp-boxscore",
        action="store_true",
        help="Include opponent raw box-score average differentials as extra features.",
    )
    parser.add_argument(
        "--include-glm-quality",
        action="store_true",
        help="Include season-level OLS team-quality differential as an extra feature.",
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
    parser.add_argument("--elo-winner-bonus", type=float, default=0.0)
    parser.add_argument("--elo-early-k-boost-games", type=int, default=0)
    parser.add_argument("--elo-early-k-multiplier", type=float, default=1.0)
    parser.add_argument("--elo-conference-reversion", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    context = load_frozen_women_context(args.data_dir)
    elo_ratings = _build_women_elo_ratings(context, args)
    feature_table = (
        build_elo_seed_tourney_features(context.results, context.seeds, elo_ratings, league="W")
        if _uses_custom_women_elo(args)
        else context.training.copy()
    )
    feature_cols = ["seed_diff", "elo_diff"]
    if args.include_tourney_elo:
        tourney_elo = compute_tournament_only_elo_ratings(context.results, seeds=context.seeds)
        feature_table = attach_secondary_elo_features(
            feature_table,
            tourney_elo,
            prefix="tourney_elo",
        )
        feature_cols.append("tourney_elo_diff")
    if args.include_elo_momentum:
        elo_momentum = compute_elo_momentum_features(
            context.regular_season,
            mid_day_cutoff=115,
            end_day_cutoff=134,
            initial_rating=WOMEN_ELO_PARAMS["initial_rating"],
            k_factor=WOMEN_ELO_PARAMS["k_factor"],
            home_advantage=WOMEN_ELO_PARAMS["home_advantage"],
            season_carryover=WOMEN_ELO_PARAMS["season_carryover"],
            scale=WOMEN_ELO_PARAMS["scale"],
            mov_alpha=WOMEN_ELO_PARAMS["mov_alpha"],
            weight_regular=WOMEN_ELO_PARAMS["weight_regular"],
            winner_bonus=args.elo_winner_bonus,
            early_k_boost_games=args.elo_early_k_boost_games,
            early_k_multiplier=args.elo_early_k_multiplier,
            team_conferences=(
                pd.read_csv(args.data_dir / "WTeamConferences.csv")
                if args.elo_conference_reversion
                else None
            ),
            conference_reversion=args.elo_conference_reversion,
        )[["Season", "TeamID", "elo_momentum"]].rename(columns={"elo_momentum": "elo"})
        feature_table = attach_secondary_elo_features(
            feature_table,
            elo_momentum,
            prefix="elo_momentum",
        )
        feature_cols.append("elo_momentum_diff")
    if args.include_pythag:
        pythag = build_team_season_summary(context.regular_season)[
            ["Season", "TeamID", "pythag_expectancy"]
        ]
        feature_table = _attach_team_scalar_feature(
            feature_table,
            pythag,
            feature_name="pythag_expectancy",
            diff_name="pythag_diff",
        )
        feature_cols.append("pythag_diff")
    if args.include_seed_elo_gap:
        feature_cols.append("seed_elo_gap_diff")
    if args.include_season_momentum:
        season_momentum = build_season_momentum_features(context.regular_season)[
            ["Season", "TeamID", "season_momentum"]
        ]
        feature_table = _attach_team_scalar_feature(
            feature_table,
            season_momentum,
            feature_name="season_momentum",
            diff_name="season_momentum_diff",
        )
        feature_cols.append("season_momentum_diff")
    if args.include_espn_four_factor:
        seasons = sorted(
            season
            for season in feature_table["Season"].drop_duplicates().astype(int).tolist()
            if season <= max(args.holdout_seasons)
        )
        espn_features = load_espn_women_four_factor_strength_features(
            espn_root=args.espn_root,
            seasons=seasons,
            regular_season_results=context.regular_season,
            team_spellings_path=args.team_spellings_path,
        )
        feature_table = _attach_team_scalar_feature(
            feature_table,
            espn_features[["Season", "TeamID", "espn_four_factor_strength"]],
            feature_name="espn_four_factor_strength",
            diff_name="espn_four_factor_strength_diff",
        )
        feature_cols.append("espn_four_factor_strength_diff")
        if args.include_espn_components:
            feature_table, espn_component_cols = _attach_espn_component_features(
                feature_table,
                espn_features,
            )
            feature_cols.extend(espn_component_cols)
    elif args.include_espn_components:
        seasons = sorted(
            season
            for season in feature_table["Season"].drop_duplicates().astype(int).tolist()
            if season <= max(args.holdout_seasons)
        )
        espn_features = load_espn_women_four_factor_strength_features(
            espn_root=args.espn_root,
            seasons=seasons,
            regular_season_results=context.regular_season,
            team_spellings_path=args.team_spellings_path,
        )
        feature_table, espn_component_cols = _attach_espn_component_features(
            feature_table,
            espn_features,
        )
        feature_cols.extend(espn_component_cols)
    if args.include_late5_form or args.include_late_bundle:
        regular_season_detailed = pd.read_csv(args.data_dir / "WRegularSeasonDetailedResults.csv")
        late5 = build_late5_form_split_features(regular_season_detailed)
        feature_table = _attach_team_scalar_feature(
            feature_table,
            late5[["Season", "TeamID", "late5_off_eff"]],
            feature_name="late5_off_eff",
            diff_name="late5_off_diff",
        )
        feature_table = _attach_team_scalar_feature(
            feature_table,
            late5[["Season", "TeamID", "late5_def_eff"]],
            feature_name="late5_def_eff",
            diff_name="late5_def_diff",
            defensive=True,
        )
        feature_cols.extend(["late5_off_diff", "late5_def_diff"])
    if args.include_opp_boxscore:
        regular_season_detailed = pd.read_csv(args.data_dir / "WRegularSeasonDetailedResults.csv")
        opp_box = build_opponent_raw_boxscore_features(regular_season_detailed)
        for feature_name, diff_name in [
            ("avg_opp_score", "avg_opp_score_diff"),
            ("avg_opp_fga", "avg_opp_fga_diff"),
            ("avg_opp_blk", "avg_opp_blk_diff"),
            ("avg_opp_pf", "avg_opp_pf_diff"),
            ("avg_opp_to", "avg_opp_to_diff"),
            ("avg_opp_stl", "avg_opp_stl_diff"),
        ]:
            feature_table = _attach_team_scalar_feature(
                feature_table,
                opp_box[["Season", "TeamID", feature_name]],
                feature_name=feature_name,
                diff_name=diff_name,
            )
            feature_cols.append(diff_name)
    if args.include_conference_rank or args.include_late_bundle:
        team_conferences = pd.read_csv(args.data_dir / "WTeamConferences.csv")
        summary_strength = build_team_season_summary(context.regular_season)[
            ["Season", "TeamID", "avg_margin"]
        ]
        conf_rank = build_conference_percentile_features(
            summary_strength,
            team_conferences,
            strength_col="avg_margin",
        )
        feature_table = _attach_team_scalar_feature(
            feature_table,
            conf_rank[["Season", "TeamID", "conf_pct_rank"]],
            feature_name="conf_pct_rank",
            diff_name="conf_pct_rank_diff",
        )
        feature_cols.append("conf_pct_rank_diff")
    if args.include_pace:
        pace = build_team_season_summary(context.regular_season)[["Season", "TeamID", "pace"]]
        feature_table = _attach_team_scalar_feature(
            feature_table,
            pace,
            feature_name="pace",
            diff_name="pace_diff",
        )
        feature_cols.append("pace_diff")
    if args.include_glm_quality:
        glm_quality = build_glm_quality_features(context.regular_season)
        feature_table = _attach_team_scalar_feature(
            feature_table,
            glm_quality,
            feature_name="glm_quality",
            diff_name="glm_quality_diff",
        )
        feature_cols.append("glm_quality_diff")
    if args.include_late_bundle:
        site_profiles = build_site_performance_features(context.regular_season)
        win_quality = build_win_quality_bin_features(context.regular_season)
        pedigree = build_program_pedigree_features(context.seeds, context.results)
        late_bundle = (
            late5.merge(site_profiles, on=["Season", "TeamID"], how="outer")
            .merge(win_quality, on=["Season", "TeamID"], how="outer")
            .merge(conf_rank, on=["Season", "TeamID"], how="outer")
            .merge(pedigree, on=["Season", "TeamID"], how="outer")
            .fillna(0.0)
        )
        feature_table = _attach_team_scalar_feature(
            feature_table,
            late_bundle[["Season", "TeamID", "road_win_pct"]],
            feature_name="road_win_pct",
            diff_name="road_win_pct_diff",
        )
        feature_table = _attach_team_scalar_feature(
            feature_table,
            late_bundle[["Season", "TeamID", "neutral_margin"]],
            feature_name="neutral_margin",
            diff_name="neutral_net_eff_diff",
        )
        feature_table = _attach_team_scalar_feature(
            feature_table,
            late_bundle[["Season", "TeamID", "close_win_pct_5"]],
            feature_name="close_win_pct_5",
            diff_name="close_win_pct_5_diff",
        )
        feature_table = _attach_team_scalar_feature(
            feature_table,
            late_bundle[["Season", "TeamID", "blowout_win_pct_15"]],
            feature_name="blowout_win_pct_15",
            diff_name="blowout_win_pct_diff",
        )
        feature_table = _attach_team_scalar_feature(
            feature_table,
            late_bundle[["Season", "TeamID", "pedigree_score"]],
            feature_name="pedigree_score",
            diff_name="pedigree_score_diff",
        )
        feature_cols.extend(
            [
                "road_win_pct_diff",
                "neutral_net_eff_diff",
                "close_win_pct_5_diff",
                "blowout_win_pct_diff",
                "pedigree_score_diff",
            ]
        )

    output_dir = args.output_dir / "w_routed_round_group"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        feature_table.to_parquet(
            output_dir / "women_routed_round_group_features.parquet",
            index=False,
        )

    summary = _validate_holdouts(
        feature_table,
        holdout_seasons=args.holdout_seasons,
        feature_cols=feature_cols,
        use_decay_weighting=args.use_decay_weighting,
        decay_base=args.decay_base,
    )
    validation_artifacts = save_validation_artifacts(summary, output_dir=output_dir / "validation")

    latest_holdout = max(args.holdout_seasons)
    infer_rows = build_seeded_submission_rows(context.seeds, season=latest_holdout)
    infer_frame = build_elo_seed_submission_features(
        infer_rows[["Season", "LowTeamID", "HighTeamID"]],
        context.seeds,
        elo_ratings,
        league="W",
    )
    if args.include_tourney_elo:
        tourney_elo = compute_tournament_only_elo_ratings(context.results, seeds=context.seeds)
        infer_frame = attach_secondary_elo_features(
            infer_frame,
            tourney_elo,
            prefix="tourney_elo",
        )
    if args.include_elo_momentum:
        elo_momentum = compute_elo_momentum_features(
            context.regular_season,
            mid_day_cutoff=115,
            end_day_cutoff=134,
            initial_rating=WOMEN_ELO_PARAMS["initial_rating"],
            k_factor=WOMEN_ELO_PARAMS["k_factor"],
            home_advantage=WOMEN_ELO_PARAMS["home_advantage"],
            season_carryover=WOMEN_ELO_PARAMS["season_carryover"],
            scale=WOMEN_ELO_PARAMS["scale"],
            mov_alpha=WOMEN_ELO_PARAMS["mov_alpha"],
            weight_regular=WOMEN_ELO_PARAMS["weight_regular"],
            winner_bonus=args.elo_winner_bonus,
            early_k_boost_games=args.elo_early_k_boost_games,
            early_k_multiplier=args.elo_early_k_multiplier,
            team_conferences=(
                pd.read_csv(args.data_dir / "WTeamConferences.csv")
                if args.elo_conference_reversion
                else None
            ),
            conference_reversion=args.elo_conference_reversion,
        )[["Season", "TeamID", "elo_momentum"]].rename(columns={"elo_momentum": "elo"})
        infer_frame = attach_secondary_elo_features(
            infer_frame,
            elo_momentum,
            prefix="elo_momentum",
        )
    if args.include_pythag:
        pythag = build_team_season_summary(context.regular_season)[
            ["Season", "TeamID", "pythag_expectancy"]
        ]
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            pythag,
            feature_name="pythag_expectancy",
            diff_name="pythag_diff",
        )
    if args.include_season_momentum:
        season_momentum = build_season_momentum_features(context.regular_season)[
            ["Season", "TeamID", "season_momentum"]
        ]
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            season_momentum,
            feature_name="season_momentum",
            diff_name="season_momentum_diff",
        )
    if args.include_espn_four_factor:
        seasons = sorted(
            season
            for season in context.seeds["Season"].drop_duplicates().astype(int).tolist()
            if season <= latest_holdout
        )
        espn_features = load_espn_women_four_factor_strength_features(
            espn_root=args.espn_root,
            seasons=seasons,
            regular_season_results=context.regular_season,
            team_spellings_path=args.team_spellings_path,
        )
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            espn_features[["Season", "TeamID", "espn_four_factor_strength"]],
            feature_name="espn_four_factor_strength",
            diff_name="espn_four_factor_strength_diff",
        )
        if args.include_espn_components:
            infer_frame, _ = _attach_espn_component_features(infer_frame, espn_features)
    elif args.include_espn_components:
        seasons = sorted(
            season
            for season in context.seeds["Season"].drop_duplicates().astype(int).tolist()
            if season <= latest_holdout
        )
        espn_features = load_espn_women_four_factor_strength_features(
            espn_root=args.espn_root,
            seasons=seasons,
            regular_season_results=context.regular_season,
            team_spellings_path=args.team_spellings_path,
        )
        infer_frame, _ = _attach_espn_component_features(infer_frame, espn_features)
    if args.include_late5_form or args.include_late_bundle:
        regular_season_detailed = pd.read_csv(args.data_dir / "WRegularSeasonDetailedResults.csv")
        late5 = build_late5_form_split_features(regular_season_detailed)
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            late5[["Season", "TeamID", "late5_off_eff"]],
            feature_name="late5_off_eff",
            diff_name="late5_off_diff",
        )
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            late5[["Season", "TeamID", "late5_def_eff"]],
            feature_name="late5_def_eff",
            diff_name="late5_def_diff",
            defensive=True,
        )
    if args.include_opp_boxscore:
        regular_season_detailed = pd.read_csv(args.data_dir / "WRegularSeasonDetailedResults.csv")
        opp_box = build_opponent_raw_boxscore_features(regular_season_detailed)
        for feature_name, diff_name in [
            ("avg_opp_score", "avg_opp_score_diff"),
            ("avg_opp_fga", "avg_opp_fga_diff"),
            ("avg_opp_blk", "avg_opp_blk_diff"),
            ("avg_opp_pf", "avg_opp_pf_diff"),
            ("avg_opp_to", "avg_opp_to_diff"),
            ("avg_opp_stl", "avg_opp_stl_diff"),
        ]:
            infer_frame = _attach_team_scalar_feature(
                infer_frame,
                opp_box[["Season", "TeamID", feature_name]],
                feature_name=feature_name,
                diff_name=diff_name,
            )
    if args.include_conference_rank or args.include_late_bundle:
        team_conferences = pd.read_csv(args.data_dir / "WTeamConferences.csv")
        summary_strength = build_team_season_summary(context.regular_season)[
            ["Season", "TeamID", "avg_margin"]
        ]
        conf_rank = build_conference_percentile_features(
            summary_strength,
            team_conferences,
            strength_col="avg_margin",
        )
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            conf_rank[["Season", "TeamID", "conf_pct_rank"]],
            feature_name="conf_pct_rank",
            diff_name="conf_pct_rank_diff",
        )
    if args.include_pace:
        pace = build_team_season_summary(context.regular_season)[["Season", "TeamID", "pace"]]
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            pace,
            feature_name="pace",
            diff_name="pace_diff",
        )
    if args.include_glm_quality:
        glm_quality = build_glm_quality_features(context.regular_season)
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            glm_quality,
            feature_name="glm_quality",
            diff_name="glm_quality_diff",
        )
    if args.include_late_bundle:
        site_profiles = build_site_performance_features(context.regular_season)
        win_quality = build_win_quality_bin_features(context.regular_season)
        pedigree = build_program_pedigree_features(context.seeds, context.results)
        late_bundle = (
            late5.merge(site_profiles, on=["Season", "TeamID"], how="outer")
            .merge(win_quality, on=["Season", "TeamID"], how="outer")
            .merge(conf_rank, on=["Season", "TeamID"], how="outer")
            .merge(pedigree, on=["Season", "TeamID"], how="outer")
            .fillna(0.0)
        )
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            late_bundle[["Season", "TeamID", "road_win_pct"]],
            feature_name="road_win_pct",
            diff_name="road_win_pct_diff",
        )
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            late_bundle[["Season", "TeamID", "neutral_margin"]],
            feature_name="neutral_margin",
            diff_name="neutral_net_eff_diff",
        )
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            late_bundle[["Season", "TeamID", "close_win_pct_5"]],
            feature_name="close_win_pct_5",
            diff_name="close_win_pct_5_diff",
        )
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            late_bundle[["Season", "TeamID", "blowout_win_pct_15"]],
            feature_name="blowout_win_pct_15",
            diff_name="blowout_win_pct_diff",
        )
        infer_frame = _attach_team_scalar_feature(
            infer_frame,
            late_bundle[["Season", "TeamID", "pedigree_score"]],
            feature_name="pedigree_score",
            diff_name="pedigree_score_diff",
        )
    infer_frame = _score_inference_frame(
        feature_table,
        infer_frame,
        season=latest_holdout,
        feature_cols=feature_cols,
        use_decay_weighting=args.use_decay_weighting,
        decay_base=args.decay_base,
    )
    infer_frame["ID"] = infer_rows["ID"].to_numpy()

    bracket = compute_bracket_diagnostics(
        context.slots,
        context.seeds,
        infer_frame[["ID", "Pred"]],
        season=latest_holdout,
        results=context.results,
        round_col="Round",
    )
    bracket_artifacts = save_bracket_artifacts(bracket, output_dir=output_dir / "bracket")

    if args.log_mlflow:
        _log_mlflow_run(args, summary, validation_artifacts, bracket_artifacts, feature_cols)

    print(summary.per_season_metrics.to_string(index=False))
    print(
        "\nOverall metrics:",
        f"flat_brier={summary.overall_flat_brier:.6f}",
        f"log_loss={summary.overall_log_loss:.6f}",
    )
    print(f"Artifacts written under {output_dir}")
    return 0


def _validate_holdouts(
    frame: pd.DataFrame,
    *,
    holdout_seasons: list[int],
    feature_cols: list[str],
    use_decay_weighting: bool = False,
    decay_base: float = 0.9,
    train_min_games: int = 50,
) -> ValidationSummary:
    work = frame.loc[frame["outcome"].notna()].copy()
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

        pred = _score_with_routed_models(
            train,
            valid,
            season=holdout_season,
            feature_cols=feature_cols,
            use_decay_weighting=use_decay_weighting,
            decay_base=decay_base,
        )
        pred_frame = pd.DataFrame(
            {
                "Season": valid["Season"].to_numpy(),
                "LowTeamID": valid["LowTeamID"].to_numpy(),
                "HighTeamID": valid["HighTeamID"].to_numpy(),
                "outcome": valid["outcome"].astype(int).to_numpy(),
                "pred": pred.to_numpy(),
                "round_group": valid["round_group"].to_numpy(),
                "Round": valid["Round"].to_numpy(),
            }
        )
        prediction_frames.append(pred_frame)
        metrics_rows.append(
            {
                "Season": holdout_season,
                "train_tourney_games": len(train),
                "valid_tourney_games": len(valid),
                "flat_brier": float(brier_score_loss(pred_frame["outcome"], pred_frame["pred"])),
                "log_loss": float(log_loss(pred_frame["outcome"], pred_frame["pred"])),
                "r1_brier": _group_brier(pred_frame, "R1"),
                "r2plus_brier": _group_brier(pred_frame, "R2+"),
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


def _score_inference_frame(
    frame: pd.DataFrame,
    infer_frame: pd.DataFrame,
    *,
    season: int,
    feature_cols: list[str],
    use_decay_weighting: bool = False,
    decay_base: float = 0.9,
) -> pd.DataFrame:
    train = frame.loc[frame["Season"] < season].copy()
    pred = _score_with_routed_models(
        train,
        infer_frame,
        season=season,
        feature_cols=feature_cols,
        use_decay_weighting=use_decay_weighting,
        decay_base=decay_base,
    )
    scored = infer_frame.copy()
    scored["Pred"] = pred
    return scored


def _score_with_routed_models(
    train_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    season: int,
    feature_cols: list[str],
    use_decay_weighting: bool = False,
    decay_base: float = 0.9,
) -> pd.Series:
    routed_preds: dict[str, pd.Series] = {}
    fallback_model = build_logistic_pipeline(feature_cols)
    fit_kwargs: dict[str, pd.Series] = {}
    if use_decay_weighting:
        fit_kwargs["model__sample_weight"] = _season_decay_weights(
            train_frame["Season"],
            decay_base=decay_base,
        )
    fallback_model.fit(train_frame[feature_cols], train_frame["outcome"].astype(int), **fit_kwargs)
    fallback_pred = pd.Series(
        fallback_model.predict_proba(target_frame[feature_cols])[:, 1],
        index=target_frame.index,
        dtype=float,
    )

    for round_group in ROUND_GROUPS:
        group_train = train_frame.loc[train_frame["round_group"] == round_group].copy()
        if group_train.empty:
            continue
        model = build_logistic_pipeline(feature_cols)
        group_fit_kwargs: dict[str, pd.Series] = {}
        if use_decay_weighting:
            group_fit_kwargs["model__sample_weight"] = _season_decay_weights(
                group_train["Season"],
                decay_base=decay_base,
            )
        model.fit(
            group_train[feature_cols],
            group_train["outcome"].astype(int),
            **group_fit_kwargs,
        )
        routed_preds[round_group] = pd.Series(
            model.predict_proba(target_frame[feature_cols])[:, 1],
            index=target_frame.index,
            dtype=float,
        )

    if not routed_preds:
        raise ValueError(f"No routed women training rows available before season {season}.")

    return route_group_predictions(
        routed_preds,
        target_frame["round_group"],
        fallback=fallback_pred,
    )


def _season_decay_weights(seasons: pd.Series, *, decay_base: float) -> pd.Series:
    latest = int(seasons.max())
    weights = seasons.astype(int).map(lambda season: float(decay_base ** (latest - int(season))))
    return weights.astype(float)


def _uses_custom_women_elo(args: argparse.Namespace) -> bool:
    return (
        args.elo_winner_bonus != 0.0
        or args.elo_early_k_boost_games > 0
        or args.elo_early_k_multiplier != 1.0
        or args.elo_conference_reversion
    )


def _build_women_elo_ratings(context, args: argparse.Namespace) -> pd.DataFrame:
    if not _uses_custom_women_elo(args):
        return context.elo_ratings

    team_conferences = (
        pd.read_csv(args.data_dir / "WTeamConferences.csv")
        if args.elo_conference_reversion
        else None
    )

    return compute_pre_tourney_elo_ratings(
        context.regular_season,
        tourney_results=context.results,
        day_cutoff=134,
        initial_rating=WOMEN_ELO_PARAMS["initial_rating"],
        k_factor=WOMEN_ELO_PARAMS["k_factor"],
        home_advantage=WOMEN_ELO_PARAMS["home_advantage"],
        season_carryover=WOMEN_ELO_PARAMS["season_carryover"],
        scale=WOMEN_ELO_PARAMS["scale"],
        mov_alpha=WOMEN_ELO_PARAMS["mov_alpha"],
        weight_regular=WOMEN_ELO_PARAMS["weight_regular"],
        weight_tourney=WOMEN_ELO_PARAMS["weight_tourney"],
        winner_bonus=args.elo_winner_bonus,
        early_k_boost_games=args.elo_early_k_boost_games,
        early_k_multiplier=args.elo_early_k_multiplier,
        team_conferences=team_conferences,
        conference_reversion=args.elo_conference_reversion,
    )


def _attach_team_scalar_feature(
    frame: pd.DataFrame,
    team_feature: pd.DataFrame,
    *,
    feature_name: str,
    diff_name: str,
    defensive: bool = False,
) -> pd.DataFrame:
    enriched = frame.merge(
        team_feature.rename(columns={"TeamID": "LowTeamID", feature_name: f"low_{feature_name}"}),
        on=["Season", "LowTeamID"],
        how="left",
        validate="many_to_one",
    ).merge(
        team_feature.rename(columns={"TeamID": "HighTeamID", feature_name: f"high_{feature_name}"}),
        on=["Season", "HighTeamID"],
        how="left",
        validate="many_to_one",
    )
    if defensive:
        enriched[diff_name] = enriched[f"high_{feature_name}"].astype(float) - enriched[
            f"low_{feature_name}"
        ].astype(float)
    else:
        enriched[diff_name] = enriched[f"low_{feature_name}"].astype(float) - enriched[
            f"high_{feature_name}"
        ].astype(float)
    return enriched


def _attach_espn_component_features(
    frame: pd.DataFrame,
    espn_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    component_specs = [
        ("espn_efg", "espn_efg_diff"),
        ("espn_tov_rate", "espn_tov_rate_diff"),
        ("espn_orb_pct", "espn_orb_pct_diff"),
        ("espn_ftr", "espn_ftr_diff"),
        ("espn_opp_efg", "espn_opp_efg_diff"),
        ("espn_opp_tov_rate", "espn_opp_tov_rate_diff"),
        ("espn_opp_orb_pct", "espn_opp_orb_pct_diff"),
        ("espn_opp_ftr", "espn_opp_ftr_diff"),
    ]
    enriched = frame
    added: list[str] = []
    for feature_name, diff_name in component_specs:
        enriched = _attach_team_scalar_feature(
            enriched,
            espn_features[["Season", "TeamID", feature_name]],
            feature_name=feature_name,
            diff_name=diff_name,
        )
        if diff_name not in added:
            added.append(diff_name)
    return enriched, added


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
) -> None:
    tags = {
        "hypothesis": (
            "A true routed women R1-vs-R2+ model may improve held-out flat Brier "
            "over the unified ARCH-04B baseline"
        ),
        "model_family": "routed_logistic_regression",
        "league": "women",
        "season_window": ",".join(str(season) for season in args.holdout_seasons),
        "depends_on": "arch-04b,women_round_group_diagnostics",
        "retest_if": "women routed round-group training definition changes",
        "leakage_audit": "passed",
    }
    if args.include_espn_four_factor:
        tags["depends_on"] += ",feature:late_feat_18_espn_four_factor_v1"
    if args.include_espn_components:
        tags["depends_on"] += ",feature:late_ext_03_espn_components_v1"
    if args.include_late5_form:
        tags["depends_on"] += ",feature:late_feat_24_late5_split_v1"
    if args.include_conference_rank:
        tags["depends_on"] += ",feature:late_feat_29_conf_pct_rank_v1"
    if args.include_pace:
        tags["depends_on"] += ",feature:cooper_feat_01_pace_v1"
    if args.include_opp_boxscore:
        tags["depends_on"] += ",feature:mh7_feat_02_opp_boxscore_v1"
    if args.include_glm_quality:
        tags["depends_on"] += ",feature:mh7_feat_01_glm_quality_v1"
    if args.use_decay_weighting:
        tags["depends_on"] += ",arch:late_arch_dw_01_v1"
    if args.include_late_bundle:
        tags["depends_on"] += ",feature:late_feat_bundle_24_27_28_29_31_v1"
    with start_tracked_run(args.run_name, tags=tags):
        mlflow.log_params(
            {
                "holdout_seasons": ",".join(str(season) for season in args.holdout_seasons),
                "features": ",".join(feature_cols),
                "round_groups": ",".join(ROUND_GROUPS),
                "include_tourney_elo": str(args.include_tourney_elo).lower(),
                "include_elo_momentum": str(args.include_elo_momentum).lower(),
                "include_pythag": str(args.include_pythag).lower(),
                "include_seed_elo_gap": str(args.include_seed_elo_gap).lower(),
                "include_season_momentum": str(args.include_season_momentum).lower(),
                "include_espn_four_factor": str(args.include_espn_four_factor).lower(),
                "include_espn_components": str(args.include_espn_components).lower(),
                "include_late5_form": str(args.include_late5_form).lower(),
                "include_conference_rank": str(args.include_conference_rank).lower(),
                "include_pace": str(args.include_pace).lower(),
                "include_opp_boxscore": str(args.include_opp_boxscore).lower(),
                "include_glm_quality": str(args.include_glm_quality).lower(),
                "include_late_bundle": str(args.include_late_bundle).lower(),
                "use_decay_weighting": str(args.use_decay_weighting).lower(),
                "decay_base": args.decay_base,
                "elo_winner_bonus": args.elo_winner_bonus,
                "elo_early_k_boost_games": args.elo_early_k_boost_games,
                "elo_early_k_multiplier": args.elo_early_k_multiplier,
                "elo_conference_reversion": str(args.elo_conference_reversion).lower(),
            }
        )
        mlflow.log_metrics(
            {
                "flat_brier": summary.overall_flat_brier,
                "log_loss": summary.overall_log_loss,
                "r1_brier_mean": float(summary.per_season_metrics["r1_brier"].dropna().mean()),
                "r2plus_brier_mean": float(
                    summary.per_season_metrics["r2plus_brier"].dropna().mean()
                ),
            }
        )
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
