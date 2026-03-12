from __future__ import annotations

import pandas as pd
from sklearn.metrics import brier_score_loss  # type: ignore[import-untyped]


def blend_predictions(
    first_preds: pd.Series | list[float],
    second_preds: pd.Series | list[float],
    *,
    first_weight: float,
) -> pd.Series:
    """Blend two probability vectors with a convex weight on the first model."""
    first = pd.Series(first_preds, dtype=float).reset_index(drop=True)
    second = pd.Series(second_preds, dtype=float).reset_index(drop=True)
    if len(first) != len(second):
        raise ValueError("Prediction vectors must have the same length to be blended.")
    return first_weight * first + (1.0 - first_weight) * second


def find_best_blend_weight(
    first_preds: pd.Series | list[float],
    second_preds: pd.Series | list[float],
    outcomes: pd.Series | list[int],
    *,
    step: float = 0.05,
) -> tuple[float, pd.DataFrame]:
    """Grid-search the best convex blend weight using flat Brier."""
    if step <= 0 or step > 1:
        raise ValueError("step must be in the interval (0, 1].")

    outcome_series = pd.Series(outcomes, dtype=int).reset_index(drop=True)
    rows: list[dict[str, float]] = []
    weight = 0.0
    while weight < 1.0 + 1e-9:
        blended = blend_predictions(first_preds, second_preds, first_weight=weight)
        rows.append(
            {
                "first_weight": round(weight, 10),
                "flat_brier": float(brier_score_loss(outcome_series, blended)),
            }
        )
        weight += step

    grid = pd.DataFrame(rows).sort_values(["flat_brier", "first_weight"]).reset_index(drop=True)
    best_weight = float(grid.iloc[0]["first_weight"])
    return best_weight, grid


def merge_prediction_frames(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    pred_cols: tuple[str, str] = ("pred_first", "pred_second"),
) -> pd.DataFrame:
    """Merge two prediction frames on canonical game identifiers."""
    required = {"Season", "LowTeamID", "HighTeamID", "outcome", "pred"}
    missing_first = required.difference(first.columns)
    missing_second = required.difference(second.columns)
    if missing_first:
        raise ValueError(
            f"First prediction frame missing required columns: {sorted(missing_first)}"
        )
    if missing_second:
        raise ValueError(
            f"Second prediction frame missing required columns: {sorted(missing_second)}"
        )

    merged = first.merge(
        second,
        on=["Season", "LowTeamID", "HighTeamID", "outcome"],
        how="inner",
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    if "round_group_first" in merged.columns:
        merged["round_group"] = merged["round_group_first"]
        merged = merged.drop(
            columns=[
                col for col in ["round_group_first", "round_group_second"] if col in merged.columns
            ]
        )
    merged = merged.rename(
        columns={
            "pred_first": pred_cols[0],
            "pred_second": pred_cols[1],
        }
    )
    return merged.sort_values(["Season", "LowTeamID", "HighTeamID"]).reset_index(drop=True)
