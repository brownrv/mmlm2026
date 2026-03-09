# Canonical NCAA Tournament Round Assignment

Round assignment must be seed-based, not date-based.

## Rule

Use:
- `assign_rounds_from_seeds(tc, seeds)` in `src/mmlm2026/round_utils.py`
- `data/tourney_round_lookup.csv`

Do not infer round from `DayNum`.

## Inputs

- `tc`: tournament results table with `Season`, `WTeamID`, `LTeamID`
- `seeds`: NCAA tournament seeds with `Season`, `TeamID`, `Seed` (e.g. `W01`, `X11a`)
- `tourney_round_lookup.csv`: columns `StrongSeed`, `WeakSeed`, `Round`, `Slot`

## Implementation (4 Steps)

1. Map `TeamID -> Seed` using `(Season, TeamID)` from `NCAATourneySeeds`.
2. Normalize seeds by stripping trailing `a` / `b` play-in suffixes.
3. Detect play-in games:
If normalized seeds are identical (e.g. `W16a` vs `W16b` -> `W16`), assign `Round = 0`.
4. Lookup `(seed1_norm, seed2_norm)` in the bidirectional dictionary built from `tourney_round_lookup.csv`.

## Behavior

- Works for men and women.
- Works across historical seasons despite schedule anomalies.
- Returns `Round = -1` if a pairing is unresolvable.

## Important Separation of Concerns

- `DayNum < 134` filters used for Elo cutoff/training windows are still valid for that purpose.
- But `DayNum` must not be used to determine tournament round.
