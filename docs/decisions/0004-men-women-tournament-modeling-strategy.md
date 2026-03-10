# ADR 0004: men-women tournament modeling strategy

Status: Accepted
Date: 2026-03-10
Owners: repo maintainers

## Context

The competition produces a single submission covering both men's and women's NCAA
tournaments. A consistent policy is needed for how models are structured, trained,
and combined for M vs W data to avoid ad-hoc per-session decisions.

## Decision

**Option A: Separate models per league, merged at submission time.**

Each league (M and W) has its own independently trained model (or ensemble). Final
submission is assembled by concatenating the M and W prediction sets.

## Alternatives Considered

- Option B: Single shared model with `league` as an explicit categorical feature.
- Option C: Separate feature pipelines feeding a shared model architecture.
- Option D: Other.

## Why This Decision

- Women's tournament has meaningfully different parity profiles than men's.
- Men's data includes Massey ordinals and coach records not available for women; a shared
  model forces either imputation or feature degradation for women.
- Women's data starts at 2010 for detailed results; a shared model inherits men's longer
  history in ways that may hurt women's calibration.
- Separate models allow league-specific hyperparameter tuning, calibration curves, and
  feature sets without compromising either league.
- Merge at submission time is low-risk: simply concatenate M and W prediction CSVs.

## Consequences

### Positive
- Full control over women's calibration independently of men's.
- Feature sets can differ between leagues without imputation hacks.
- Each league's model can be validated and diagnosed independently.

### Negative
- Doubles the number of models and experiments to maintain.
- No cross-league signal sharing (e.g. a shared strength metric) — accepted tradeoff.

## Dependencies

- league:men
- league:women

## Invalidated by

- If competition splits into separate M/W leaderboards (submission format changes).
- If women's dataset gains comparable depth to men's (Massey ordinals, coach records).

## Related Experiments

- docs/experiments/experiment-log.md
- PLAN-002 §4.2, §9.1

## Follow-up Actions

- [x] Decision recorded; PLAN-002 §4.2 updated to reflect Option A as active policy.
- [ ] Add MLflow runs with `league` tag for each affected experiment family.
