from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import optuna
import pandas as pd

from mmlm2026.features.elo import pregame_expected_winner_probability
from mmlm2026.round_utils import assign_rounds_from_seeds

TuningMode = Literal["replication", "generalization"]


@dataclass(frozen=True)
class EloParams:
    initial_rating: float
    k_factor: float
    home_advantage: float
    season_carryover: float
    scale: float
    mov_alpha: float
    weight_regular: float
    weight_tourney: float


@dataclass(frozen=True)
class GameRow:
    season: int
    winner: int
    loser: int
    winner_score: float
    loser_score: float
    winner_location: str
    is_tourney: bool


@dataclass(frozen=True)
class StudyArtifacts:
    best_params: EloParams
    best_value: float
    objective_seasons: list[int]
    report_seasons: list[int]
    include_playins: bool
    mode: TuningMode
    boundary_hits: dict[str, bool]
    output_dir: Path


def default_objective_seasons(mode: TuningMode) -> list[int]:
    if mode == "replication":
        return [2022, 2023, 2024, 2025]
    return [2015, 2016, 2017, 2018, 2019, 2021]


def default_report_seasons(mode: TuningMode) -> list[int]:
    if mode == "replication":
        return [2022, 2023, 2024, 2025]
    return [2022, 2023, 2024, 2025]


def default_include_playins(mode: TuningMode) -> bool:
    return mode == "replication"


def broad_search_ranges() -> dict[str, tuple[float, float]]:
    return {
        "initial_rating": (800.0, 2000.0),
        "k_factor": (20.0, 200.0),
        "home_advantage": (30.0, 200.0),
        "season_carryover": (0.5, 1.0),
        "scale": (200.0, 2000.0),
        "mov_alpha": (0.0, 50.0),
        "weight_regular": (0.5, 1.5),
        "weight_tourney": (0.5, 1.5),
    }


def narrow_ranges(
    best: EloParams,
    *,
    base_ranges: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    min_half_width = {
        "initial_rating": 50.0,
        "k_factor": 10.0,
        "home_advantage": 10.0,
        "season_carryover": 0.01,
        "scale": 100.0,
        "mov_alpha": 1.0,
        "weight_regular": 0.1,
        "weight_tourney": 0.1,
    }
    narrowed: dict[str, tuple[float, float]] = {}
    for key, (lower, upper) in base_ranges.items():
        value = getattr(best, key)
        delta = max(abs(value) * 0.25, min_half_width[key])
        narrowed[key] = (max(lower, value - delta), min(upper, value + delta))
    return narrowed


def boundary_hits(
    best: EloParams,
    *,
    ranges: dict[str, tuple[float, float]],
    atol: dict[str, float] | None = None,
) -> dict[str, bool]:
    default_atol = {
        "initial_rating": 1.0,
        "k_factor": 1e-6,
        "home_advantage": 1e-6,
        "season_carryover": 1e-4,
        "scale": 1.0,
        "mov_alpha": 1e-4,
        "weight_regular": 1e-4,
        "weight_tourney": 1e-4,
    }
    tolerance = atol or default_atol
    hits: dict[str, bool] = {}
    for key, (lower, upper) in ranges.items():
        value = getattr(best, key)
        hits[key] = abs(value - lower) <= tolerance[key] or abs(value - upper) <= tolerance[key]
    return hits


def prepare_game_rows(
    *,
    regular: pd.DataFrame,
    tourney: pd.DataFrame,
    seeds: pd.DataFrame | None = None,
    include_playins: bool,
) -> list[GameRow]:
    required_regular = {"Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"}
    required_tourney = {"Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore"}
    missing_regular = required_regular.difference(regular.columns)
    missing_tourney = required_tourney.difference(tourney.columns)
    if missing_regular:
        raise ValueError(
            f"Regular season results missing required columns: {sorted(missing_regular)}"
        )
    if missing_tourney:
        raise ValueError(f"Tournament results missing required columns: {sorted(missing_tourney)}")

    regular_rows = regular.assign(is_tourney=0).copy()
    tourney_rows = tourney.copy()
    if not include_playins:
        if seeds is None:
            raise ValueError("Seeds are required when include_playins is False.")
        tourney_rounds = assign_rounds_from_seeds(
            tourney_rows[["Season", "WTeamID", "LTeamID"]].copy(),
            seeds[["Season", "Seed", "TeamID"]].copy(),
        )
        keep_mask = tourney_rounds["Round"] > 0
        tourney_rows = tourney_rows.loc[keep_mask].copy()
    tourney_rows["WLoc"] = "N"
    tourney_rows["is_tourney"] = 1
    all_games = pd.concat([regular_rows, tourney_rows], ignore_index=True).sort_values(
        ["Season", "DayNum", "is_tourney"]
    )
    return [
        GameRow(
            season=int(row["Season"]),
            winner=int(row["WTeamID"]),
            loser=int(row["LTeamID"]),
            winner_score=float(row["WScore"]),
            loser_score=float(row["LScore"]),
            winner_location=str(row["WLoc"]),
            is_tourney=bool(row["is_tourney"]),
        )
        for _, row in all_games.iterrows()
    ]


def objective_brier(
    *,
    params: EloParams,
    games: list[GameRow],
    objective_seasons: list[int],
) -> float:
    ratings: dict[int, float] = {}
    current_season: int | None = None
    losses: list[float] = []
    objective_season_set = set(objective_seasons)

    for row in games:
        season = row.season
        if current_season is None:
            current_season = season
        elif season != current_season:
            ratings = {
                team: params.season_carryover * rating
                + (1.0 - params.season_carryover) * params.initial_rating
                for team, rating in ratings.items()
            }
            current_season = season

        winner_rating = ratings.setdefault(row.winner, params.initial_rating)
        loser_rating = ratings.setdefault(row.loser, params.initial_rating)
        exp_w = pregame_expected_winner_probability(
            winner_rating,
            loser_rating,
            winner_location=row.winner_location,
            scale=params.scale,
            home_advantage=params.home_advantage,
        )
        if row.is_tourney and season in objective_season_set:
            losses.append((1.0 - exp_w) ** 2)

        mov_mult = 1.0
        if params.mov_alpha > 0.0:
            mov_mult = 1.0 + max(0.0, row.winner_score - row.loser_score) / params.mov_alpha
        weight = params.weight_tourney if row.is_tourney else params.weight_regular
        delta = weight * params.k_factor * mov_mult * (1.0 - exp_w)
        ratings[row.winner] = winner_rating + delta
        ratings[row.loser] = max(loser_rating - delta, 1.0)

    if not losses:
        raise ValueError("No tournament objective games found for the requested seasons.")
    return float(sum(losses) / len(losses))


def run_study(
    *,
    study_name: str,
    storage_path: Path,
    games: list[GameRow],
    objective_seasons: list[int],
    seed: int,
    n_trials: int,
    ranges: dict[str, tuple[float, float]],
    show_progress_bar: bool = False,
) -> EloParams:
    sampler = optuna.samplers.TPESampler(seed=seed)
    storage_uri = f"sqlite:///{storage_path.resolve().as_posix()}"
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=study_name,
        storage=storage_uri,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = EloParams(
            initial_rating=trial.suggest_float("initial_rating", *ranges["initial_rating"]),
            k_factor=trial.suggest_float("k_factor", *ranges["k_factor"]),
            home_advantage=trial.suggest_float("home_advantage", *ranges["home_advantage"]),
            season_carryover=trial.suggest_float(
                "season_carryover",
                *ranges["season_carryover"],
            ),
            scale=trial.suggest_float("scale", *ranges["scale"]),
            mov_alpha=trial.suggest_float("mov_alpha", *ranges["mov_alpha"]),
            weight_regular=trial.suggest_float("weight_regular", *ranges["weight_regular"]),
            weight_tourney=trial.suggest_float("weight_tourney", *ranges["weight_tourney"]),
        )
        return objective_brier(
            params=params,
            games=games,
            objective_seasons=objective_seasons,
        )

    remaining_trials = max(n_trials - len(study.trials), 0)
    print(
        f"{study_name}: existing_trials={len(study.trials)} "
        f"target_trials={n_trials} remaining_trials={remaining_trials}"
    )
    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            show_progress_bar=show_progress_bar,
        )
    best = study.best_params
    return EloParams(
        initial_rating=float(best["initial_rating"]),
        k_factor=float(best["k_factor"]),
        home_advantage=float(best["home_advantage"]),
        season_carryover=float(best["season_carryover"]),
        scale=float(best["scale"]),
        mov_alpha=float(best["mov_alpha"]),
        weight_regular=float(best["weight_regular"]),
        weight_tourney=float(best["weight_tourney"]),
    )


def save_study_artifacts(
    *,
    output_dir: Path,
    study_name: str,
    storage_path: Path,
    best_params: EloParams,
    ranges: dict[str, tuple[float, float]],
    objective_seasons: list[int],
    report_seasons: list[int],
    include_playins: bool,
    mode: TuningMode,
) -> StudyArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_uri = f"sqlite:///{storage_path.resolve().as_posix()}"
    study = optuna.load_study(study_name=study_name, storage=storage_uri)
    hits = boundary_hits(best_params, ranges=ranges)
    trials = study.trials_dataframe()
    trials.to_parquet(output_dir / f"{study_name}_trials.parquet", index=False)
    summary = {
        "study_name": study_name,
        "best_value": float(study.best_value),
        "best_params": asdict(best_params),
        "ranges": {key: list(value) for key, value in ranges.items()},
        "objective_seasons": objective_seasons,
        "report_seasons": report_seasons,
        "include_playins": include_playins,
        "mode": mode,
        "boundary_hits": hits,
        "n_trials": len(study.trials),
    }
    (output_dir / f"{study_name}_summary.json").write_text(json.dumps(summary, indent=2))
    return StudyArtifacts(
        best_params=best_params,
        best_value=float(study.best_value),
        objective_seasons=objective_seasons,
        report_seasons=report_seasons,
        include_playins=include_playins,
        mode=mode,
        boundary_hits=hits,
        output_dir=output_dir,
    )
