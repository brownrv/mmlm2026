# ADR 0002: NCAA round assignment uses seed-pair lookup

Status: Accepted
Date: 2026-03-10
Owners: repo maintainers

## Context

Tournament `DayNum` is not a stable proxy for NCAA round across all seasons and
both leagues. Scheduling anomalies and historical differences can break day-based
round inference.

## Decision

Use canonical seed-pair lookup for round assignment:

- implementation: `assign_rounds_from_seeds(tc, seeds)` in `src/mmlm2026/round_utils.py`
- lookup table: `data/tourney_round_lookup.csv`

Play-in handling rule:
- if normalized seeds are equal (e.g. `W16a` vs `W16b` -> `W16`), assign `Round = 0`.

## Alternatives Considered

- Infer round from `DayNum` ranges
- Infer round from slots only without seed normalization
- Maintain separate men/women day-to-round rules

## Why This Decision

- Deterministic across men's and women's tournaments.
- Robust to season-specific schedule variation (including known anomalies).
- Keeps round logic independent from training-window logic.

## Consequences

### Positive
- Consistent round labels across all supported seasons.
- Fewer hidden assumptions in feature engineering.
- Clear separation between round assignment and Elo date cutoffs.

### Negative
- Requires maintaining canonical lookup CSV.
- Requires seed availability and normalization in data prep.

## Dependencies

- `data/tourney_round_lookup.csv`
- `MNCAATourneySeeds.csv`
- `WNCAATourneySeeds.csv`
- `src/mmlm2026/round_utils.py`

## Invalidated by

- Competition data format changes that remove/rename seed semantics.
- Evidence that seed-pair round mapping is inconsistent with official brackets.
- A future official Kaggle field that provides authoritative round labels across seasons.

## Related Experiments

- docs/experiments/experiment-log.md
- docs/data/TOURNEY_ROUND_ASSIGNMENT.md

## Follow-up Actions

- [ ] Add unit tests for representative seed-pair round cases and play-in detection.
- [ ] Add data-quality check for unresolved round assignments (`Round = -1`).
