# Experiment Log

Use this file as a lightweight chronological index of notable experiments.

---
## YYYY-MM-DD — <experiment title>

Status: Proposed | Running | Completed | Revisit | Retired

Hypothesis:
- <one sentence>

Dependencies:
- rating_model:<version>
- sim_engine:<version>

MLflow:
- Run name: <name>
- Run ID: <id>

Result:
- <one or two bullets>

Re-test if:
- <trigger condition>

Related:
- docs/experiments/<file>.md
- docs/decisions/<file>.md

---
## 2026-03-09 — Infrastructure complete; no model experiments run yet

Status: Completed

Hypothesis:
- N/A — this is an infrastructure milestone marker, not a model experiment.

Dependencies:
- N/A

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Phases 1-12 of the original setup plan are complete (see `docs/roadmaps/EXECUTION_STATUS.md`).
- Stage 1 infrastructure implemented: data loaders (`src/mmlm2026/data/loaders.py`),
  evaluation splits (`src/mmlm2026/evaluation/splits.py`), submission writer
  (`src/mmlm2026/submission/writer.py`), MLflow helper
  (`src/mmlm2026/utils/mlflow_tracking.py`), round assignment utility
  (`src/mmlm2026/round_utils.py`).
- No predictive model experiments have been run as of this date.

Re-test if:
- N/A

Related:
- docs/roadmaps/EXECUTION_STATUS.md
- docs/decisions/0001-experiment-ledger-is-mlflow.md

---
## 2026-03-10 — Gate 0: submission validation utility implemented

Status: Completed

Hypothesis:
- A dedicated submission validator will catch schema and ID issues before upload and reduce last-minute submission failures.

Dependencies:
- plan:002_gate0
- submission_writer:v1

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added `src/mmlm2026/submission/validation.py` with reusable validation for schema, ID format, duplicate IDs, prediction range, and optional sample-submission coverage checks.
- Added `scripts/validate_submission.py` as the Gate 0 CLI entry point referenced by PLAN-002.
- Added tests in `tests/test_submission_validation.py`; `uv run pytest` passed locally.

Re-test if:
- Kaggle submission schema changes.
- The project changes clipping bounds away from `[0.025, 0.975]`.
- Sample-submission handling becomes league- or stage-specific.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — Gate 0: seed-diff baseline run plumbing implemented

Status: Completed

Hypothesis:
- A dedicated seed-diff baseline runner will make ARCH-01 and ARCH-02 reproducible and close the remaining Gate 0 plumbing gap before actual MLflow baseline runs.

Dependencies:
- plan:002_gate0
- validate_cv:v1
- bracket_dp:v1

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added `src/mmlm2026/features/baseline.py` to build played-game seed-diff tournament features in canonical `LowTeamID` orientation with `round_group` and `outcome`.
- Added `scripts/run_seed_diff_baseline.py` to build baseline features from raw Kaggle tournament files, run leave-season-out validation, generate bracket diagnostics for the latest holdout season, and optionally log to MLflow.
- Added `tests/test_baseline_features.py`; `uv run pytest`, `uv run ruff check .`, and `uv run mypy src` passed locally.

Re-test if:
- Seed-diff feature orientation or sign convention changes.
- ARCH-01/02 move to a different baseline model family.
- Stage 2 baseline inference requires additional fields beyond the current canonical baseline feature set.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — Gate 0: bracket diagnostics utility implemented

Status: Completed

Hypothesis:
- Deterministic bracket DP diagnostics will expose which matchup probabilities matter most in bracket space without changing the primary flat-Brier selection metric.

Dependencies:
- plan:002_gate0
- validate_cv:v1
- round_assignment:seed_pair_lookup

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added `src/mmlm2026/evaluation/bracket.py` with slot recursion, play-in slot support, per-pair play probabilities, per-slot win probabilities, and optional realized-game bucket/round summaries.
- Added `scripts/bracket_dp.py` as the CLI for season-specific bracket diagnostics from slots, seeds, matchup probabilities, and optional tournament results.
- Added tests in `tests/test_bracket.py`; `uv run pytest`, `uv run ruff check .`, and `uv run mypy src` passed locally.

Re-test if:
- Kaggle slot graph conventions change.
- Play-in slot handling changes for men or women.
- The project changes bucket thresholds or diagnostic summary expectations.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — Gate 0: leave-season-out CV validation utility implemented

Status: Completed

Hypothesis:
- A reusable leave-season-out validation CLI will make baseline and model-family comparisons reproducible and enforce the plan's temporal evaluation rules.

Dependencies:
- plan:002_gate0
- submission_validation:v1
- mlflow_tracking:v1

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added `src/mmlm2026/evaluation/validation.py` with strict pre-holdout training, per-season flat Brier/log-loss, optional `R1` and `R2+` diagnostics, calibration-table generation, and reliability-diagram export.
- Added `scripts/validate_cv.py` as the Gate 0 CLI entry point with optional MLflow logging and required `leakage_audit` tagging when MLflow logging is enabled.
- Added tests in `tests/test_validation.py`; `uv run pytest`, `uv run ruff check .`, and `uv run mypy src` passed locally. `scikit-learn` imports are locally documented with `import-untyped` suppressions because the package is untyped in this environment.

Re-test if:
- The validation split policy changes away from strict pre-holdout training.
- The repo adopts a different baseline model family or calibration workflow.
- `scikit-learn` ships typed packages and the local `import-untyped` suppressions can be removed.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md
