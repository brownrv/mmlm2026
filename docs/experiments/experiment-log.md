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
## 2026-03-10 — Gate 1: Feature Phase B builders implemented

Status: Completed

Hypothesis:
- Reusable Phase B season-level feature builders will let primary-model experiments move beyond seed and Elo baselines without forcing feature logic to live in notebooks or one-off scripts.

Dependencies:
- plan:002_gate1
- feature:elo_v1
- round_assignment:seed_pair_lookup

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added `src/mmlm2026/features/phase_b.py` with season-level builders for adjusted efficiency, strength of schedule, recent form, and Massey consensus features.
- Added `tests/test_phase_b_features.py`; `uv run pytest`, `uv run ruff check .`, and `uv run mypy src` passed locally after the Phase B implementation.
- Gate 1 feature infrastructure now covers the Phase B feature families called for in PLAN-002, so the next work can move to primary-model runs instead of additional baseline plumbing.

Re-test if:
- The pre-tournament cutoff policy changes away from `DayNum < 134` / `RankingDayNum <= 133`.
- Phase B feature definitions change materially before ARCH-05 and ARCH-06.
- A selected model family needs matchup-level transformations beyond the current season-level builders.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

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
## 2026-03-10 — VAL-01: 2025 holdout sanity check (men)

Status: Completed

Hypothesis:
- The current best men’s baseline (`ARCH-01`) generalizes from the 2023–2024 validation window to the unseen 2025 tournament without a material degradation.

Dependencies:
- plan:002_gate1
- arch-01
- validation:leave_season_out_v1

MLflow:
- Run name: val-01-2025-holdout-men
- Run ID: c3733a50efc7491586f3060e0dc317b0

Result:
- 2025 holdout produced `flat_brier = 0.151642` and `log_loss = 0.469892`.
- Mean round-group diagnostics: `r1_brier = 0.135707`, `r2plus_brier = 0.168092`.
- This is better than the earlier 2023–2024 validation average from `ARCH-01` (`flat_brier = 0.197508`), so there is no evidence of overfitting from the current men’s baseline.

Re-test if:
- A new men’s baseline overtakes `ARCH-01`.
- The validation window or leakage policy changes.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — VAL-01: 2025 holdout sanity check (women)

Status: Completed

Hypothesis:
- The current best women’s baseline (`ARCH-04`) generalizes from the 2023–2024 validation window to the unseen 2025 tournament without a material degradation.

Dependencies:
- plan:002_gate1
- arch-04
- validation:leave_season_out_v1

MLflow:
- Run name: val-01-2025-holdout-women
- Run ID: 84f848e9634340f1adf0233d36ded46f

Result:
- 2025 holdout produced `flat_brier = 0.113286` and `log_loss = 0.361987`.
- Mean round-group diagnostics: `r1_brier = 0.087981`, `r2plus_brier = 0.139406`.
- This is better than the earlier 2023–2024 validation average from `ARCH-04` (`flat_brier = 0.137357`), so there is no evidence of overfitting from the current women’s baseline.

Re-test if:
- A new women’s baseline overtakes `ARCH-04`.
- The validation window or leakage policy changes.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-03: Elo + seed logistic baseline (men)

Status: Completed

Hypothesis:
- End-of-regular-season Elo adds signal beyond seed difference for the men’s tournament baseline.

Dependencies:
- plan:002_gate1
- feature:seed_diff_v1
- feature:elo_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-03-elo-seed-men
- Run ID: ed65e8bcc7e04961a97bbccd4e868c71

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.197613` and `log_loss = 0.580609`.
- Mean round-group diagnostics: `r1_brier = 0.185721`, `r2plus_brier = 0.209889`.
- Relative to `ARCH-01` (`flat_brier = 0.197508`), Elo did not improve the men’s baseline on this validation window.

Re-test if:
- Elo hyperparameters (`day_cutoff`, `k_factor`, `home_advantage`) change.
- Men’s baseline feature set expands beyond `seed_diff` and `elo_diff`.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-04: Elo + seed logistic baseline (women)

Status: Completed

Hypothesis:
- End-of-regular-season Elo adds signal beyond seed difference for the women’s tournament baseline.

Dependencies:
- plan:002_gate1
- feature:seed_diff_v1
- feature:elo_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-04-elo-seed-women
- Run ID: c4929b684eea490faaca353aefbb5e46

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.137357` and `log_loss = 0.426218`.
- Mean round-group diagnostics: `r1_brier = 0.103508`, `r2plus_brier = 0.172297`.
- Relative to `ARCH-02` (`flat_brier = 0.140573`), Elo improved the women’s baseline on this validation window.

Re-test if:
- Elo hyperparameters (`day_cutoff`, `k_factor`, `home_advantage`) change.
- Women’s baseline feature set expands beyond `seed_diff` and `elo_diff`.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-01: seed-diff logistic baseline (men)

Status: Completed

Hypothesis:
- Seed difference alone is a strong men’s tournament baseline and should beat the naive 0.5 floor with reasonable Round of 64 calibration.

Dependencies:
- plan:002_gate1
- feature:seed_diff_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-01-seed-diff-men
- Run ID: c0ef5d21cbf44cc185edc4ec7b920b43

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.197508` and `log_loss = 0.579247`.
- Mean round-group diagnostics: `r1_brier = 0.182200`, `r2plus_brier = 0.213310`.
- Per-season flat Brier: 2023 `0.204203`, 2024 `0.190814`.

Re-test if:
- Seed parsing or `seed_diff` sign convention changes.
- Round-group routing or bracket diagnostics change materially.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-02: seed-diff logistic baseline (women)

Status: Completed

Hypothesis:
- Seed difference alone is an even stronger women’s tournament baseline than men’s and should provide a solid floor for later feature additions.

Dependencies:
- plan:002_gate1
- feature:seed_diff_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-02-seed-diff-women
- Run ID: 809fa43caaf24469b2a2b68cf3afbfc9

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.140573` and `log_loss = 0.432945`.
- Mean round-group diagnostics: `r1_brier = 0.104118`, `r2plus_brier = 0.178204`.
- Per-season flat Brier: 2023 `0.169764`, 2024 `0.111382`.

Re-test if:
- Seed parsing or `seed_diff` sign convention changes.
- Women’s tournament seed dynamics shift materially in future seasons.

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
