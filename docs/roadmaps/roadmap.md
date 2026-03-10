# Roadmap (Canonical Working Roadmap)

This is the canonical roadmap file for active project planning.

## Near-Term

### Current Priorities

- Finalize baseline data loading and validation split pipeline.
- Produce first reproducible baseline model run with MLflow tracking.
- Standardize submission writer checks (schema + probability clipping).
- Improve baseline rating pipeline.
- Validate simulation assumptions.
- Strengthen calibration checks.
- Improve experiment logging discipline.
- Add submission validation checks.

### Next Up

- Add reusable matchup probability tooling.
- Standardize men/women model comparison workflow.
- Link MLflow runs to Git commit SHA automatically.

## Mid-Term

- Expand feature families (ratings, matchup interactions, tempo proxies).
- Add simulation-driven features and tournament-round sensitivity.
- Improve calibration monitoring and post-hoc correction strategy.

## Revisit Queue / Parking Lot

- Ensemble/stacking policy after robust single-model baselines.
- Automatic trigger-to-issue creation for invalidated assumptions.
- Explore player-level or score-distribution modeling.
- Investigate round-specific model families.
