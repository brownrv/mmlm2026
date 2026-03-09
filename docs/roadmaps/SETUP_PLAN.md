# MMLM2026 Repository Setup Plan

Author: Senior Data Scientist / ML Engineer Guidance Purpose: Establish
a high-performance Kaggle research environment with strong experimental
discipline, reproducibility, and AI-agent collaboration.

This plan is designed so both Codex and Claude Code can read and execute
the setup steps.

------------------------------------------------------------------------

# Core Philosophy

This repository is a Kaggle research lab, not just a code repository.

Primary goals: 1. Maximize leaderboard performance 2. Maintain rigorous
experimental discipline 3. Ensure reproducibility 4. Enable fast
iteration 5. Capture long-term research memory

Key systems:

  Question                     Source of Truth
  ---------------------------- ------------------------------------
  What changed?                Git
  Did it help?                 MLflow
  Why did we believe this?     docs/decisions
  What experiments were run?   docs/experiments
  What should be revisited?    GitHub Issues + retest-triggers.md

------------------------------------------------------------------------

# Phase 1 --- Create GitHub Repository

Create a new repository named:

mmlm2026

Recommended settings: - Python .gitignore - README.md - Public or
Private depending on preference - No copied code from previous repos

Mission statement (README):

mmlm2026 is a clean-room Kaggle research repository for March Machine
Learning Mania 2026. Its purpose is to maximize leaderboard performance
while maintaining strong experimental discipline, reproducibility, and
long-term learning.

------------------------------------------------------------------------

# Phase 2 --- Local Clone and Base Structure

Clone repo locally:

git clone https://github.com/`<username>`{=html}/mmlm2026.git cd
mmlm2026

Create project structure:

mmlm2026/ ├─ .github/ │ ├─ workflows/ │ └─ ISSUE_TEMPLATE/ ├─ data/ │ ├─
raw/ │ ├─ interim/ │ ├─ processed/ │ ├─ features/ │ ├─ submissions/ │ └─
manifests/ ├─ docs/ │ ├─ decisions/ │ ├─ experiments/ │ └─ roadmaps/ ├─
notebooks/ ├─ scripts/ ├─ src/ │ └─ mmlm2026/ │ ├─ data/ │ ├─ features/
│ ├─ models/ │ ├─ evaluation/ │ ├─ simulation/ │ ├─ submission/ │ └─
utils/ ├─ tests/ ├─ AGENTS.md ├─ CLAUDE.md ├─ pyproject.toml ├─
README.md └─ .gitignore

------------------------------------------------------------------------

# Phase 3 --- Python Environment

Use modern Python tooling.

Stack: - uv - pytest - ruff - mypy - MLflow

Example commands:

uv sync uv run pytest uv run ruff check . uv run ruff format . uv run
mypy src

Initial dependencies: - pandas - numpy - scikit-learn - mlflow -
jupyter - matplotlib - pyarrow - pyyaml

------------------------------------------------------------------------

# Phase 4 --- Agent Configuration

Add two instruction files.

AGENTS.md → Codex and general agent policy CLAUDE.md → Claude Code
project memory

They should define: - repository purpose - notebook policy - experiment
logging policy - Git workflow expectations - rules for modifying data -
instructions for updating documentation - instructions for creating
Issues when assumptions change

------------------------------------------------------------------------

# Phase 5 --- Research Memory System

Create documentation system.

docs/ decisions/ experiments/ roadmaps/

Files:

docs/decisions/ 0000-template.md

docs/experiments/ experiment-template.md experiment-log.md
retest-triggers.md

docs/roadmaps/ roadmap.md

Purpose:

Ensure ideas, experiments, and assumptions are never lost.

------------------------------------------------------------------------

# Phase 6 --- Data Discipline

Use lightweight dataset versioning.

data/raw Immutable Kaggle downloads

data/interim Cleaned source data

data/processed Stable derived tables

data/features Model-ready datasets

data/manifests Dataset metadata files

Example manifest:

name: regular_season_features_v1 created_at: YYYY-MM-DD source_files: -
MRegularSeasonDetailedResults.csv code_version: `<git sha>`{=html}
notes: baseline feature build

------------------------------------------------------------------------

# Phase 7 --- MLflow Experiment Tracking

Use local MLflow.

Log: - run name - git SHA - branch - feature set version - validation
window - model type - hyperparameters - metrics - artifacts -
hypothesis - dependencies - retest triggers

Suggested tags:

hypothesis depends_on retest_if feature_set validation_scheme
model_family league

------------------------------------------------------------------------

# Phase 8 --- GitHub Issues and Project Board

Use Issues for: - ideas - experiment follow-ups - bugs - retests -
roadmap tasks

Recommended labels:

experiment idea retest data feature-engineering modeling simulation
submission-strategy infra docs

Project board columns:

Inbox Ready Running Review Blocked Revisit Later Done

------------------------------------------------------------------------

# Phase 9 --- Continuous Integration

Create GitHub Actions workflow.

Run on push and PR.

Minimum tasks:

uv sync ruff check mypy src pytest

------------------------------------------------------------------------

# Phase 10 --- Notebook Policy

Notebooks allowed for:

EDA visualization feature exploration comparison experiments

Reusable logic must move to src.

Example naming:

01_eda_kaggle_data.ipynb 10_seed_diff_baseline.ipynb
20_team_strength_ratings.ipynb 30_matchup_simulation.ipynb

------------------------------------------------------------------------

# Phase 11 --- Modeling Roadmap

Stage 1 --- Infrastructure data loaders validation splits submission
writer MLflow helpers

Stage 2 --- Baselines seed difference model Elo ratings team strength
baseline probability calibration

Stage 3 --- Tournament modeling matchup probability simulation
round-sensitive features

Stage 4 --- Ensemble strategy model stacking manual overrides
simulation-weighted predictions

------------------------------------------------------------------------

# Phase 12 --- Human + Agent Workflow

Loop:

1.  Create Issue
2.  Write hypothesis
3.  Explore in notebook
4.  Move code to src
5.  Add tests
6.  Run MLflow experiment
7.  Update experiment log
8.  Update decisions if beliefs change
9.  Add retest triggers
10. Commit

------------------------------------------------------------------------

# First Week Plan

Day 1 Create repo Create folder structure Setup uv environment Add
AGENTS.md and CLAUDE.md Add CI workflow

Day 2 Setup docs system Configure GitHub labels and project board

Day 3 Add MLflow utilities Add baseline data loaders

Day 4 Load Kaggle data Create dataset manifests Perform EDA

Day 5 Build first baseline model Log experiment Record findings

------------------------------------------------------------------------

# Final Principle

Never trust:

-   a single experiment
-   a leaderboard improvement without explanation
-   a feature not logged in MLflow
-   a modeling decision not recorded in docs

Great Kaggle work compounds over time. This system ensures knowledge
accumulates instead of being lost.
