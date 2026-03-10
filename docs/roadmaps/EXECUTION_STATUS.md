# Execution Status

Last updated: 2026-03-10

---

## PLAN-002: Competition Attack Plan

Status: Active
Source plan: `docs/roadmaps/PLAN-002-competition-attack-plan.md`

Note: Evaluation policy was revised on 2026-03-10. Flat Brier on held-out played tournament games is now the primary selection metric; bracket-aware and `R1`/`R2+` analyses are supporting diagnostics.

| Gate | Status | Target Date |
|---|---|---|
| Gate 0 — Infrastructure Ready | Not started | 2026-03-11 |
| Gate 1 — Baselines Established | Not started | 2026-03-13 |
| Gate 2 — Primary Models Ready | Not started | 2026-03-16 |
| Gate 3 — Ensemble and Final Submission | Not started | 2026-03-18 |
| Hard Deadline | Not started | 2026-03-19 16:00 UTC |

**ADR 0004 resolved (2026-03-10):** Option A — separate models per league. Round-group separation (R1 vs R2+) is an open experiment axis (ARCH-RG-01 through ARCH-RG-06).

---

> **Note:** `docs/roadmaps/SETUP_PLAN.md` has been archived to
> `docs/roadmaps/archive/SETUP_PLAN.md` (2026-03-09). All 12 phases were complete.
>
> **Note:** `docs/roadmaps/PLAN-001-repo-improvements.md` was completed and archived to
> `docs/roadmaps/archive/PLAN-001-repo-improvements.md` on 2026-03-10.

---

## PLAN-001: Repository Improvement and Agent UX Hardening

Status: Complete

Source plan: `docs/roadmaps/archive/PLAN-001-repo-improvements.md` (archived 2026-03-10)

### Phase 1 — Critical Blockers

| Task | Status | Notes |
|---|---|---|
| P1-1: Add `mlruns/` to `.gitignore` | Complete | `mlruns/` added under project-specific ignores; `git check-ignore -v mlruns/` confirms ignore. |
| P1-2: Add AI session restart reminder to `MASTER_CHECKLIST.md` | Complete | Reminder added in “After Reboot / New Session”. |
| P1-3: Document stdlib-only constraint in CI step name | Complete | CI step renamed to explicitly call out stdlib-only pre-uv execution. |

### Phase 2 — Documentation Hygiene

| Task | Status | Notes |
|---|---|---|
| P2-1: Consolidate duplicate roadmap files | Complete | Merged roadmap content into `docs/roadmaps/roadmap.md`; `docs/roadmap.md` now redirects; roadmap note updated in `CLAUDE.md` and `AGENTS.md`. |
| P2-2: Add `league` MLflow tag to `AGENTS.md` and `CLAUDE.md` | Complete | `league` added to recommended MLflow tags in both files. |
| P2-3: Add MLflow UI and backup guidance to `AGENTS.md` | Complete | Added `## MLflow Operations` in `AGENTS.md` and mirrored operational note in `CLAUDE.md`. |
| P2-4: Mark completed follow-up actions in ADR 0001 | Complete | Both follow-up actions are checked with implementation references. |
| P2-5: Add audience headers to `CLAUDE.md` and `AGENTS.md` | Complete | Audience HTML comments added as first line in both files. |
| P2-6: Add initial experiment log stub entry | Complete | Added dated infrastructure milestone entry to `docs/experiments/experiment-log.md`. |
| P2-7: Create M+W modeling strategy ADR skeleton | Complete | ADR 0004 is now accepted: Option A — separate models per league, merged at submission time. |

### Phase 3 — Agent UX Improvements

| Task | Status | Notes |
|---|---|---|
| P3-1: Narrow memory discipline hook scope | Complete | Hook now targets `src/mmlm2026/` only and message text was updated accordingly. |
| P3-2: Add `ruff format --check` to CI | Complete | Added `Ruff format check` step directly after lint in CI workflow. |
| P3-3: Enable `disallow_untyped_defs` in mypy | Complete | Set `disallow_untyped_defs = true` and validated `uv run mypy src` passes. |
| P3-4: Expand `HUMAN_AGENT_WORKFLOW.md` | Complete | Added delegation guidance, verification checklist, and session-start context template. |
| P3-5: Add usage example to `mlflow_tracking.py` | Complete | Expanded `start_tracked_run()` docstring with full recommended tags including `league`. |

---

## Original Setup Plan — Phases 1-12

Source plan: `docs/roadmaps/archive/SETUP_PLAN.md` (archived 2026-03-09)

### Phase Status (1-12)

| Phase | Status | Evidence |
|---|---|---|
| Phase 1 - Create GitHub Repository | Complete (local verification) | Repo is `mmlm2026`; mission statement in `README.md` matches setup plan language. |
| Phase 2 - Local Clone and Base Structure | Complete | Required folders/files exist: `.github/workflows`, `.github/ISSUE_TEMPLATE`, `data/*`, `docs/{decisions,experiments,roadmaps}`, `notebooks/`, `scripts/`, `src/mmlm2026/*`, `tests/`, `AGENTS.md`, `CLAUDE.md`, `pyproject.toml`, `README.md`, `.gitignore`. |
| Phase 3 - Python Environment | Complete | `pyproject.toml` includes required stack and dependencies (`uv` workflow, `pytest`, `ruff`, `mypy`, `mlflow`, `pandas`, `numpy`, `scikit-learn`, `jupyter`, `matplotlib`, `pyarrow`, `pyyaml`, plus `pre-commit`). Standard commands are documented and wired via `scripts/dev.ps1`. |
| Phase 4 - Agent Configuration | Complete | `AGENTS.md` and `CLAUDE.md` both present and aligned on repository purpose, notebook policy, experiment logging, data immutability, issue creation, and documentation update expectations. |
| Phase 5 - Research Memory System | Complete | `docs/decisions/`, `docs/experiments/`, and `docs/roadmaps/` are populated with templates/log files; added `docs/roadmaps/roadmap.md`. |
| Phase 6 - Data Discipline | Complete | `data/` layout matches plan and includes manifest template: `data/manifests/manifest-template.yaml`. |
| Phase 7 - MLflow Experiment Tracking | Complete | MLflow policy is documented; added reusable helper `src/mmlm2026/utils/mlflow_tracking.py`; ADR 0001 establishes MLflow as experiment ledger. |
| Phase 8 - GitHub Issues and Project Board | Partial (repo artifacts complete) | Issue template exists (`.github/ISSUE_TEMPLATE/experiment_followup.md`); added `docs/roadmaps/GITHUB_PROJECT_SETUP.md` with labels/board spec. Actual label/project board creation must be done in GitHub UI/API. |
| Phase 9 - Continuous Integration | Complete | Added `.github/workflows/ci.yml` running `uv sync --frozen`, `ruff check`, `mypy src`, and `pytest` on push/PR. |
| Phase 10 - Notebook Policy | Complete | Notebook policy codified in `AGENTS.md`, `CLAUDE.md`, and `docs/NOTEBOOK_POLICY.md`. |
| Phase 11 - Modeling Roadmap | Complete | Added staged roadmap in `docs/roadmaps/MODELING_ROADMAP.md`; stage-1 scaffolding implemented in `src/mmlm2026/{data,evaluation,submission,utils}` with initial tests. |
| Phase 12 - Human + Agent Workflow | Complete | Workflow documented in `docs/roadmaps/HUMAN_AGENT_WORKFLOW.md` and mirrored in agent instructions. |

### Notes

- Phase 1 GitHub-hosted settings (visibility, template choices) are not directly verifiable from local files alone.
- Phase 8 includes manual GitHub-side steps (labels/board) that cannot be created from local files alone.
- Based on repository contents, Phases 1-12 are implemented with the above caveats.
