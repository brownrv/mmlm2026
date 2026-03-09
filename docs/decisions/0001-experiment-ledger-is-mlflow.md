# ADR 0001: MLflow is the primary experiment ledger

Status: Accepted  
Date: 2026-03-09  
Owners: repo maintainers

## Context
This project needs a reliable way to compare experiments across models, features,
validation windows, and submission strategies. Chat history and memory are not a
durable or queryable source of record.

## Decision
MLflow is the canonical ledger for experiment runs, metrics, params, and artifacts.
Git is the canonical history for code changes. Markdown docs in `docs/` are the
canonical history for decisions, assumptions, and retest triggers.

## Alternatives Considered
- Keep experiment notes only in notebooks
- Keep history only in chat sessions
- Use commit messages as the only record of why a change was made

## Why This Decision
Each tool answers a different question:
- Git: what changed
- MLflow: did it help
- Docs: why we believed it and when to revisit it

## Consequences
### Positive
- Better long-term reproducibility
- Easier retroactive analysis
- Cleaner separation of code, metrics, and reasoning

### Negative
- Slightly more process overhead
- Requires discipline to keep notes up to date

## Dependencies
- mlflow:enabled
- git:used_for_all_code_changes
- docs:maintained

## Invalidated by
- A future tool that clearly replaces MLflow for run tracking
- Team decision to consolidate into another experiment platform

## Related Experiments
- docs/experiments/experiment-log.md

## Follow-up Actions
- [ ] Add git SHA and branch tags to all runs
- [ ] Create a reusable MLflow logging helper
