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
