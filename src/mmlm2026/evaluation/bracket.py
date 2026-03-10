from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# scikit-learn does not expose typed packages in this environment.
from sklearn.metrics import brier_score_loss  # type: ignore[import-untyped]


@dataclass(frozen=True)
class BracketArtifacts:
    """Paths to saved bracket-diagnostics artifacts."""

    play_prob_path: Path
    slot_team_prob_path: Path
    bucket_summary_path: Path | None = None
    round_summary_path: Path | None = None


@dataclass(frozen=True)
class BracketDiagnostics:
    """Structured outputs for bracket diagnostics."""

    play_probabilities: pd.DataFrame
    slot_team_probabilities: pd.DataFrame
    bucket_summary: pd.DataFrame | None = None
    round_summary: pd.DataFrame | None = None


def compute_bracket_diagnostics(
    slots: pd.DataFrame,
    seeds: pd.DataFrame,
    matchup_probs: pd.DataFrame,
    *,
    season: int,
    results: pd.DataFrame | None = None,
    pred_col: str = "Pred",
    id_col: str = "ID",
    round_col: str = "Round",
) -> BracketDiagnostics:
    """Compute per-pair play probabilities and optional realized-game diagnostics."""
    season_slots = slots.loc[slots["Season"] == season].copy()
    season_seeds = seeds.loc[seeds["Season"] == season].copy()
    if season_slots.empty:
        raise ValueError(f"No slot rows found for season {season}.")
    if season_seeds.empty:
        raise ValueError(f"No seed rows found for season {season}.")

    slot_lookup = {
        row["Slot"]: (str(row["StrongSeed"]), str(row["WeakSeed"]))
        for _, row in season_slots.iterrows()
    }
    seed_lookup = {str(row["Seed"]): int(row["TeamID"]) for _, row in season_seeds.iterrows()}
    prob_lookup = _build_matchup_probability_lookup(
        matchup_probs,
        season,
        id_col=id_col,
        pred_col=pred_col,
    )

    play_prob: dict[tuple[int, int], float] = defaultdict(float)
    slot_team_probs: dict[str, dict[int, float]] = {}

    def resolve_source(source: str) -> dict[int, float]:
        if source in slot_team_probs:
            return slot_team_probs[source]
        if source in slot_lookup:
            strong_source, weak_source = slot_lookup[source]
            left = resolve_source(strong_source)
            right = resolve_source(weak_source)
            slot_team_probs[source] = _resolve_slot(left, right, prob_lookup, play_prob)
            return slot_team_probs[source]
        if source in seed_lookup:
            return {seed_lookup[source]: 1.0}
        raise ValueError(f"Unresolvable slot or seed source '{source}' for season {season}.")

    for slot in season_slots["Slot"].tolist():
        resolve_source(str(slot))

    play_probabilities = pd.DataFrame(
        [
            {
                "Season": season,
                "LowTeamID": low_team,
                "HighTeamID": high_team,
                "play_prob": probability,
            }
            for (low_team, high_team), probability in sorted(play_prob.items())
        ]
    )
    slot_team_probabilities = pd.DataFrame(
        [
            {
                "Season": season,
                "Slot": slot,
                "TeamID": team_id,
                "win_prob": probability,
            }
            for slot, team_probs in sorted(slot_team_probs.items())
            for team_id, probability in sorted(team_probs.items())
        ]
    )

    bucket_summary = None
    round_summary = None
    if results is not None:
        bucket_summary, round_summary = summarize_realized_games(
            results,
            play_probabilities,
            matchup_probs,
            season=season,
            pred_col=pred_col,
            id_col=id_col,
            round_col=round_col,
        )

    return BracketDiagnostics(
        play_probabilities=play_probabilities,
        slot_team_probabilities=slot_team_probabilities,
        bucket_summary=bucket_summary,
        round_summary=round_summary,
    )


def save_bracket_artifacts(
    diagnostics: BracketDiagnostics,
    *,
    output_dir: str | Path,
) -> BracketArtifacts:
    """Persist bracket-diagnostics outputs to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    play_prob_path = out_dir / "play_probabilities.csv"
    slot_team_prob_path = out_dir / "slot_team_probabilities.csv"
    diagnostics.play_probabilities.to_csv(play_prob_path, index=False)
    diagnostics.slot_team_probabilities.to_csv(slot_team_prob_path, index=False)

    bucket_summary_path: Path | None = None
    round_summary_path: Path | None = None
    if diagnostics.bucket_summary is not None:
        bucket_summary_path = out_dir / "bucket_summary.csv"
        diagnostics.bucket_summary.to_csv(bucket_summary_path, index=False)
    if diagnostics.round_summary is not None:
        round_summary_path = out_dir / "round_summary.csv"
        diagnostics.round_summary.to_csv(round_summary_path, index=False)

    return BracketArtifacts(
        play_prob_path=play_prob_path,
        slot_team_prob_path=slot_team_prob_path,
        bucket_summary_path=bucket_summary_path,
        round_summary_path=round_summary_path,
    )


def summarize_realized_games(
    results: pd.DataFrame,
    play_probabilities: pd.DataFrame,
    matchup_probs: pd.DataFrame,
    *,
    season: int,
    pred_col: str = "Pred",
    id_col: str = "ID",
    round_col: str = "Round",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Summarize played games by play-probability bucket and optional round."""
    season_results = results.loc[results["Season"] == season].copy()
    if season_results.empty:
        raise ValueError(f"No realized results found for season {season}.")

    prob_lookup = _build_matchup_probability_lookup(
        matchup_probs,
        season,
        id_col=id_col,
        pred_col=pred_col,
    )
    play_lookup = {
        (int(row["LowTeamID"]), int(row["HighTeamID"])): float(row["play_prob"])
        for _, row in play_probabilities.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for _, row in season_results.iterrows():
        low_team = int(min(row["WTeamID"], row["LTeamID"]))
        high_team = int(max(row["WTeamID"], row["LTeamID"]))
        outcome = 1 if int(row["WTeamID"]) == low_team else 0
        pred = prob_lookup[(low_team, high_team)]
        play_prob = play_lookup.get((low_team, high_team), 0.0)
        rows.append(
            {
                "Season": season,
                "LowTeamID": low_team,
                "HighTeamID": high_team,
                "pred": pred,
                "outcome": outcome,
                "play_prob": play_prob,
                "bucket": _bucket_from_play_prob(play_prob),
                round_col: row[round_col] if round_col in season_results.columns else None,
            }
        )

    realized = pd.DataFrame(rows)
    bucket_summary = _summarize_groups(realized, "bucket")

    round_summary = None
    if round_col in realized.columns and realized[round_col].notna().any():
        round_summary = _summarize_groups(realized, round_col)

    return bucket_summary, round_summary


def _resolve_slot(
    left: dict[int, float],
    right: dict[int, float],
    prob_lookup: dict[tuple[int, int], float],
    play_prob: dict[tuple[int, int], float],
) -> dict[int, float]:
    team_probs: dict[int, float] = defaultdict(float)

    for left_team, left_prob in left.items():
        for right_team, right_prob in right.items():
            matchup_play_prob = left_prob * right_prob
            low_team, high_team = sorted((left_team, right_team))
            play_prob[(low_team, high_team)] += matchup_play_prob

            left_win_prob = _lookup_matchup_probability(left_team, right_team, prob_lookup)
            right_win_prob = 1.0 - left_win_prob

            team_probs[left_team] += matchup_play_prob * left_win_prob
            team_probs[right_team] += matchup_play_prob * right_win_prob

    return dict(team_probs)


def _build_matchup_probability_lookup(
    matchup_probs: pd.DataFrame,
    season: int,
    *,
    id_col: str,
    pred_col: str,
) -> dict[tuple[int, int], float]:
    required = {id_col, pred_col}
    missing = required.difference(matchup_probs.columns)
    if missing:
        raise ValueError(f"Matchup probabilities missing required columns: {sorted(missing)}")

    season_probs = matchup_probs.loc[
        matchup_probs[id_col].astype(str).str.startswith(f"{season}_")
    ].copy()
    if season_probs.empty:
        raise ValueError(f"No matchup probabilities found for season {season}.")

    lookup: dict[tuple[int, int], float] = {}
    for _, row in season_probs.iterrows():
        season_value, low_team, high_team = _parse_submission_id(str(row[id_col]))
        if season_value != season:
            continue
        lookup[(low_team, high_team)] = float(row[pred_col])
    return lookup


def _lookup_matchup_probability(
    team_a: int,
    team_b: int,
    prob_lookup: dict[tuple[int, int], float],
) -> float:
    low_team, high_team = sorted((team_a, team_b))
    try:
        low_team_win_prob = prob_lookup[(low_team, high_team)]
    except KeyError as exc:
        raise ValueError(f"Missing matchup probability for pair {(low_team, high_team)}.") from exc
    if team_a == low_team:
        return low_team_win_prob
    return 1.0 - low_team_win_prob


def _parse_submission_id(submission_id: str) -> tuple[int, int, int]:
    season_str, low_str, high_str = submission_id.split("_")
    return int(season_str), int(low_str), int(high_str)


def _bucket_from_play_prob(play_prob: float) -> str:
    if play_prob >= 0.99:
        return "definite"
    if play_prob >= 0.30:
        return "very_likely"
    if play_prob >= 0.10:
        return "likely"
    if play_prob >= 0.03:
        return "plausible"
    return "remote"


def _summarize_groups(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value, group in frame.groupby(group_col, dropna=False):
        rows.append(
            {
                group_col: group_value,
                "games": len(group),
                "mean_play_prob": float(group["play_prob"].mean()),
                "flat_brier": float(brier_score_loss(group["outcome"], group["pred"])),
            }
        )
    return pd.DataFrame(rows)
