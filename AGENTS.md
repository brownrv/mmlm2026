<!-- Audience: Codex and general agents. Claude Code additionally reads CLAUDE.md. -->
# AGENTS.md

## Project Overview
This repository contains models and data pipelines for the Kaggle competition
"March Machine Learning Mania", which predicts the probability that
Team A beats Team B in NCAA tournament matchups.

The project focuses on:
- reproducible experiments
- modular feature engineering
- MLflow experiment tracking
- Kaggle submission generation
- documented assumptions and decision history

Primary language: Python
Primary environment: Python 3.11

---

## Repository Structure

```text
repo/
├─ src/                    # Main Python package
│  └─ mmlm2026/
├─ notebooks/              # Exploratory work and analysis
├─ scripts/                # CLI utilities and maintenance scripts
├─ data/
│  ├─ raw/                 # Original external data (never modified)
│  ├─ interim/             # Intermediate outputs from cleaning and joins
│  ├─ processed/           # Cleaned / structured datasets
│  ├─ features/            # Feature tables used by models
│  ├─ submissions/         # Kaggle submission files
│  └─ manifests/           # Data version and metadata manifests
├─ docs/
│  ├─ decisions/           # ADR-style decision records
│  ├─ experiments/         # Experiment notes and research logs
│  └─ roadmaps/            # Roadmap documents (canonical: docs/roadmaps/roadmap.md)
└─ tests/                  # Unit tests
```

---

## Development Rules

### Code location
- Production logic belongs in `src/`
- Notebooks are for exploration and analysis
- Reusable logic should be moved out of notebooks into `src/`
- Never leave durable production logic only in notebooks

### Code style
- Follow PEP 8
- Prefer small focused functions
- Add docstrings to public functions
- Use type hints when practical
- Prefer explicit names over compact clever code

### Imports
Prefer absolute imports, for example:

```python
from mmlm2026.features.seed_features import build_seed_features
```

---

## Data Rules

### Raw data is immutable
Never modify files in `data/raw/`.

Current Kaggle raw snapshot is stored at:
- `data/raw/march-machine-learning-mania-2026/`

### Interim data
Intermediate outputs from cleaning and joins go in `data/interim/`.

### Processed data
Cleaned and structured datasets go in `data/processed/`.

### Feature tables
Model-ready datasets belong in `data/features/`.

### Submission artifacts
Generated competition submissions belong in `data/submissions/`.

### Manifests
Data versions and metadata manifests belong in `data/manifests/`.

### Relationship reference
Use `docs/data/RELATIONSHIP_DIAGRAM.md` for join keys and table relationships.
Use `docs/data/TOURNEY_ROUND_ASSIGNMENT.md` for NCAA round assignment rules.
Use `docs/data/TEAM_SPELLINGS_POLICY.md` for canonical team-name mapping rules.

### NCAA round assignment
- Determine tournament rounds from normalized seed-pair lookup via `data/tourney_round_lookup.csv`.
- Canonical implementation is `assign_rounds_from_seeds(tc, seeds)` in `src/mmlm2026/round_utils.py`.
- Do not infer NCAA tournament round from `DayNum`.
- `DayNum < 134` filtering for Elo cutoffs is a separate purpose and remains valid.

### Team spellings source of truth
- Prefer `data/TeamSpellings.csv` as the canonical team-spelling mapping.
- Treat Kaggle `MTeamSpellings.csv` and `WTeamSpellings.csv` as upstream sources, not final mapping tables.
- If Kaggle adds new spellings, merge missing entries into `data/TeamSpellings.csv` to keep it a superset.

---

## Experiment Tracking

Experiments are tracked with MLflow.

Every significant run should log:
- run name
- params
- metrics
- artifacts
- git commit SHA
- branch name
- dataset or feature version
- short hypothesis

Recommended tags:
- `hypothesis`
- `depends_on`
- `retest_if`
- `model_family`
- `season_window`
- `league`

---

## MLflow Operations

Default tracking URI: `./mlruns/` (relative to project root; gitignored).

Launch the MLflow UI:
```bash
uv run mlflow ui
```
Then open `http://localhost:5000` to browse runs, compare metrics, and download artifacts.

Backup: before final competition submission, snapshot `mlruns/` to a separate backup
location (external drive, cloud storage, or private branch). Do not commit `mlruns/`.

---

## Decision and Experiment Memory

Stable project decisions belong in `docs/decisions/`.

Experiment notes, failed attempts, caveats, and retest triggers belong in
`docs/experiments/`.

When an assumption changes, agents should search these folders for impacted items.

Operational checklist source of truth:
- `MASTER_CHECKLIST.md` at repo root.
- When workflows/tools/policies change, update `MASTER_CHECKLIST.md` in the same PR.

### Dependency tagging pattern
Use consistent markers in decision and experiment notes:

- `Dependencies:`
- `Invalidated by:`
- `Re-test if:`

Example:
- `Dependencies: rating_model:v2, sim_engine:v1`
- `Re-test if: score_distribution changes`

---

## Kaggle Submission Rules

Submission format:

```csv
ID,Pred
YYYY_1101_1203,0.63
YYYY_1102_1304,0.48
```

Where:
- `ID = Season_LowTeamID_HighTeamID` (lower TeamID must be first)
- `Pred = probability that the lower TeamID team wins`

Example season-specific IDs:
- `2024_1101_1203`
- `2025_1102_1304`

Never submit probabilities of exactly 0 or 1.

Clip probabilities unless there is a deliberate reason not to:
- `0.025 <= p <= 0.975`

---

## Development Commands

Install:
```bash
uv sync
```

Test:
```bash
uv run pytest
```

Lint:
```bash
uv run ruff check .
```

Format:
```bash
uv run ruff format .
```

Type check:
```bash
uv run mypy src
```

Create experiment note:
```bash
uv run python scripts/new_experiment.py "<title>" --owner "<owner>"
```

Create decision record:
```bash
uv run python scripts/new_decision.py "<title>" --owners "<owners>"
```

Memory discipline check (manual run):
```bash
uv run python scripts/check_memory_update.py
```

Changed-file policy check (manual run):
```bash
uv run python scripts/check_changed_file_policies.py --base <base_sha> --head <head_sha>
```

Install pre-commit hooks:
```bash
uv run pre-commit install
```

Run pre-commit hooks:
```bash
uv run pre-commit run --all-files
```

Pre-commit workflow:
1. Run `uv run pre-commit install` once per clone.
2. Run `uv run pre-commit run --all-files` before opening a PR.
3. Let the git hook run on every commit and stage any auto-fixes it applies.

---

## Agent Editing Guidelines

Agents should:
- prefer editing `.py` files over notebooks
- avoid modifying generated data
- avoid large refactors unless requested
- write small, reviewable commits
- add brief comments for non-obvious logic
- update experiment notes when a meaningful test is run
- update or create a decision record when a stable project decision is made
- create a follow-up issue or TODO when assumptions change and retesting is needed
- if a new idea is promising but incomplete, create an Issue so it is not lost in chat

Agents should not:
- rewrite working pipelines unnecessarily
- modify raw data
- silently change evaluation assumptions
- leave major experiment conclusions only in chat history

---

## Preferred Workflow

When implementing a new idea:
1. Explore in a notebook
2. Move reusable code into `src/`
3. Add tests
4. Track experiment in MLflow
5. Record findings in `docs/experiments/`
6. Generate submission only after validation

When a working assumption becomes stable:
1. Create or update a decision record in `docs/decisions/`
2. Link related experiments
3. Note what would invalidate the decision

---

## Notes

This repository prioritizes:
- reproducibility
- modular pipelines
- clear feature engineering
- durable experiment memory
- easy retroactive analysis of decisions and results
