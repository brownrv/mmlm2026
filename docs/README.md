# Docs Overview

This folder stores the durable project memory that chat sessions should not be trusted to hold.

## Subfolders

- `decisions/` — stable architecture, modeling, and workflow decisions
- `experiments/` — hypothesis logs, run summaries, failures, caveats, and retest triggers

## Quick rules

- Put **why** and **assumptions** here, not only in commit messages
- Put **metrics and artifacts** in MLflow
- Put **code changes** in Git
- Put **follow-up retests** in issues or explicit TODOs

## Recommended flow

1. Run experiment
2. Log run in MLflow
3. Summarize findings in `experiments/`
4. If it becomes a stable project choice, create or update a record in `decisions/`
