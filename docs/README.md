# Docs Overview

This folder stores the durable project memory that chat sessions should not be trusted to hold.

## Subfolders

- `decisions/` — stable architecture, modeling, and workflow decisions
- `experiments/` — hypothesis logs, run summaries, failures, caveats, and retest triggers
- `data/` — dataset structure references, schema notes, and relationship diagrams

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

## Data modeling references

- `MASTER_CHECKLIST.md` is the one-stop operational reminder checklist.
- `docs/data/RELATIONSHIP_DIAGRAM.md` contains the 2026 Kaggle data ER diagram and join keys.
- `docs/data/FILE_CATALOG.md` summarizes all competition files, keys, and usage roles.
- `docs/data/TOURNEY_ROUND_ASSIGNMENT.md` defines canonical NCAA round assignment from seed pairs.
- `docs/data/TEAM_SPELLINGS_POLICY.md` defines canonical team-spellings source-of-truth rules.
- `docs/roadmaps/RESEARCH_MEMORY_AUTOMATION.md` documents current memory-system automation and manual responsibilities.
- `docs/COMPETITION_OPERATIONS.md` captures timeline/limits/submission operations and update workflow.
