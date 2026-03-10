# ADR 0004: men-women tournament modeling strategy

Status: Proposed
Date: 2026-03-10
Owners: repo maintainers

## Context

The competition produces a single submission covering both men's and women's NCAA
tournaments. A consistent policy is needed for how models are structured, trained,
and combined for M vs W data to avoid ad-hoc per-session decisions.

## Decision

**[HUMAN DECISION REQUIRED]** Choose one of the following options and fill in below.

- Option A: Separate models per league, merged at submission time.
- Option B: Single shared model with `league` as an explicit categorical feature.
- Option C: Separate feature pipelines feeding a shared model architecture.
- Option D: Other (describe).

Chosen option: ___

Rationale: ___

## Alternatives Considered

- Option A: Separate models per league, merged at submission time.
- Option B: Single shared model with `league` as an explicit categorical feature.
- Option C: Separate feature pipelines feeding a shared model architecture.
- Option D: Other (describe).

## Why This Decision

To be completed after human strategy selection.

## Consequences
### Positive
- To be completed.

### Negative
- To be completed.

## Dependencies

- league:men
- league:women

## Invalidated by

- If competition splits into separate M/W leaderboards.

## Related Experiments

- docs/experiments/experiment-log.md

## Follow-up Actions

- [ ] Add baseline experiment design aligned to chosen option.
- [ ] Add MLflow runs with `league` tag for each affected experiment family.
