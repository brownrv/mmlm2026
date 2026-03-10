# Competition Operations Guide

Operational reference for March Machine Learning Mania 2026.

Primary source: `docs/march-machine-learning-mania-2026 - Overview and Data and Rules.docx`

## Submission Mechanics

- Combined competition for men's and women's tournaments.
- Submission format:
  - `ID`: `Season_LowTeamID_HighTeamID`
  - `Pred`: probability that lower TeamID team wins
- Predict all possible matchups listed in sample submission files.
- Select one final submission explicitly before deadline.

## Evaluation

- Metric: Brier score.
- Operationally equivalent to MSE on binary outcomes in this context.

## Timeline (2026 Competition)

- Start: February 19, 2026
- Final submission deadline: March 19, 2026 4PM UTC
- Tournament scoring refresh window: March 19 - April 6, 2026

Always confirm dates on the Kaggle competition page before final submissions.

## Limits and Governance

- Max team size: 5
- Max submissions per day: 5
- Final judging uses one selected submission
- Data usage/license in competition rules: CC-BY 4.0 (per rules summary)

## Data Update Operations

- Kaggle may release updated current-season files near tournament start.
- On each update:
  1. Refresh raw files in `data/raw/march-machine-learning-mania-2026/`
  2. Update `data/manifests/` metadata
  3. Re-run feature/data validation checks
  4. Re-run baseline experiments and log changes in MLflow/docs

## Project-Specific Guardrails

- Do not derive NCAA round from `DayNum`; use seed-pair lookup policy.
- Use `data/TeamSpellings.csv` as canonical name mapping.
- Keep probabilities clipped unless intentionally changed and documented.
