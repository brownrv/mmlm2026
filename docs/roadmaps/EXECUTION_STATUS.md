# Execution Status

Last updated: 2026-03-13

---

## PLAN-002: Competition Attack Plan

Status: Active
Source plan: `docs/roadmaps/PLAN-002-competition-attack-plan.md`

Note: Evaluation policy was revised on 2026-03-10. Flat Brier on held-out played tournament games is now the primary selection metric; bracket-aware and `R1`/`R2+` analyses are supporting diagnostics.

| Gate | Status | Target Date |
|---|---|---|
| Gate 0 — Infrastructure Ready | Complete | 2026-03-11 |
| Gate 1 — Baselines Established | Complete | 2026-03-13 |
| Gate 2 — Primary Models Ready | Complete | 2026-03-16 |
| Gate 3 — Ensemble and Final Submission | In progress | 2026-03-18 |
| Hard Deadline | Not started | 2026-03-19 16:00 UTC |

**ADR 0004 resolved (2026-03-10):** Option A — separate models per league. Round-group separation (R1 vs R2+) is an open experiment axis (ARCH-RG-01 through ARCH-RG-06).

**Gate 3 freeze (2026-03-12):**
- Men leader frozen to generalization-tuned men reference margin model (`337d3b992b884dbb800c078561d37622`; prior 2025 sanity check on the same reference path `a11c60d33cde4fd68f7852fc65dda1db`)
- Women leader frozen to routed women + ESPN four-factor model (`late-rate-02-women-espn-four-factor`; 2023-2024 `flat_brier = 0.130427`, 2025 sanity check run `val-01-2025-holdout-women-late-rate-02`)

Progress note (2026-03-13): Gate 0 infrastructure is complete. Gate 1 is also complete: ARCH-01 through ARCH-04 are executed and logged in MLflow (`c0ef5d21cbf44cc185edc4ec7b920b43`, `809fa43caaf24469b2a2b68cf3afbfc9`, `ed65e8bcc7e04961a97bbccd4e868c71`, `c4929b684eea490faaca353aefbb5e46`), the 2025 holdout sanity check is complete (`c3733a50efc7491586f3060e0dc317b0`, `84f848e9634340f1adf0233d36ded46f`), and Feature Phase B builders are implemented and verified. Gate 2 is now complete as an evaluation gate: ARCH-05/06 (`ffd0b86de56e40b7ac6a13e4990420c1`, `d7fa5c4793f4497ba8bb7d135591159b`), ARCH-07/08 histogram GBT (`0cfe2f69719944bcb79502464af946b6`, `98c0ca0b08c246409b782aaf627d7641`), ARCH-07/08 XGBoost retries (`087a77668be3416b99645578b2978e7a`, `bed6b72c7159465282218c492713bd6c`), VAL-02 calibration audits (`bcfa7d9e02e04600a95f214fbd1c2fd5`, `28a78f11880b42d88c6dae21b07ee7f6`, `553e266635914bcc832d3efb94538623`), COMBO-01/02 simple blends (`9b313ac531e044a182029a82f3fef182`, `d13e89d9a38448f9ab01dd76e20fc879`), FEAT-11 (`d06d9567991346639c1ce31c2e22502d`, `922d8f5438c341f985e7c692d423f43a`), FEAT-12 (`54e0cb19318840aa9fdf01b29c4fc94c`, `5279d3fd507544898087160d521463fc`), FEAT-13 (`674cc9811ca148efb72b9dbeaa799076`), FEAT-15 (`f57fb223f785496b8fa80c259be57115`), VAL-05 women reference replication (`44361f54bf444ba5a906238538af0cba`), VAL-05 women alpha reopen (`1ab9f3bd1cb549d0a0c7a3898bb2a385`), VAL-05 women reference replication v2 with local Elo (`a175a91ebe204a448b7895a8daf5e82d`), VAL-05 women reference replication with generalization Elo (`fdf6e7b90d07447ab6624b51d3be06ca`), VAL-05 men reference margin regression (`252ec46a6a874810b9e42237f7b23b1c`), VAL-05 men reference margin regression v2 (`db98ed1646cd400ead567ae4b6469a9a`), VAL-05 men reference margin regression v3 with local Elo (`f160c1daa6da4f94a1b3e9df6e2f308f`), VAL-05 men reference margin regression with generalization Elo (`337d3b992b884dbb800c078561d37622`), and ARCH-04B tuned Elo replacement (`ba0a02b8c8bc404089bdd2de7fe9a917`) challengers are complete. FEAT-12 improved on FEAT-11, especially for women, while FEAT-13 and FEAT-15 did not improve on FEAT-12. The fuller women reference replication was promising but still trailed the old baseline; local and generalization-tuned women Elo did not improve that path, and reopening alpha improved it to `0.137246` but still left it behind `ARCH-04B`. The men reference path improved first with local Elo (`0.195985`) and then again with generalization-tuned Elo (`0.195566`), giving men a slightly stronger frozen leader on the 2023–2024 held-out window. A narrow men round-group challenger (`VAL-06`) then tested separate `R1` and `R2+` calibration states on top of the frozen men model and confirmed a real split in the learned parameters, but still scored `flat_brier = 0.197124` and did not beat the frozen leader. The first true late-queue routed men model (`LATE-ARCH-RG-07`) then trained separate `R1` and `R2+` margin-regression paths and routed them at validation/inference time, but still scored `flat_brier = 0.197102`, so the frozen men leader remains unchanged. The first `LATE-RATE-01` slice then added a ridge-regularized margin strength rating from detailed regular-season results into the men reference pipeline, but it still scored `flat_brier = 0.196103`, so it also failed to beat the frozen men leader. The second `LATE-RATE-01` slice then added a leakage-safe ESPN-derived four-factor strength feature on top of that ridge-strength path, and it nearly tied the leader at `flat_brier = 0.195652` but still fell short of `0.195566`, so it also does not advance. The follow-up `LATE-FEAT-19` slice then added ESPN-derived rotation-stability proxies on top of the same path, but it degraded the held-out window to `flat_brier = 0.196864`, so those player/rotation continuity features also do not advance in the unified men reference model. The follow-up `LATE-FEAT-21` slice then added a separate tournament-only Elo signal on top of the frozen men and women leaders, but it scored `flat_brier = 0.196078` for men and `0.136614` for women, so it did not beat either frozen leader. The next low-effort Elo-derived challenger, `LATE-FEAT-22`, then added Elo momentum from `DayNum 115` to the end of the regular season and slightly improved on the tournament-only Elo men slice at `flat_brier = 0.195764`, but it still did not beat the frozen men leader at `0.195566`; the same feature degraded the women challenger path to `flat_brier = 0.138076`, so it also does not advance. The next low-effort score-derived challenger, `LATE-FEAT-26`, then added Pythagorean expectancy and produced another men near-miss at `flat_brier = 0.195654`, but it still fell short of `0.195566`; the same feature improved on several recent women late-feature challengers at `0.133332` but still remained behind the routed women leader at `0.131950`, so it also does not advance. The follow-up `LATE-FEAT-23` slice then added seed-Elo gap and materially hurt the men challenger path at `flat_brier = 0.197100`; the same feature nearly matched the routed women leader at `0.132013` but still missed `0.131950` by `0.000063`, so it also does not advance. The planned low-effort bundle of `LATE-FEAT-22`, `LATE-FEAT-26`, and `LATE-FEAT-23` then performed worse than the strongest single low-effort slices: the men bundle scored `flat_brier = 0.196494`, and the women bundle scored `0.138380`, so the three derived features are not interacting constructively in the current frozen leader families. The follow-up `LATE-FEAT-25` slice then added full-season momentum and still missed in both leagues: the men challenger scored `0.196529`, and the women challenger scored `0.132935`, which improved on some women late-feature slices but remained behind the routed women leader `0.131950`. A matching narrow women challenger (`VAL-07`) then fit targeted temperature scaling for `very_likely` and `R2+` games on top of the frozen women model; it found only a mild `R2+` adjustment, reverted `very_likely` to the global temperature, and scored `flat_brier = 0.134298`, which did not beat `ARCH-04B`. A final narrow women structural challenger (`ARCH-04C`) then tested `ARCH-04B` plus `women_hca_adj_qg_diff`, but it degraded the held-out window to `flat_brier = 0.140032`, reinforcing that the frozen women baseline should remain unchanged. The first true late-queue routed women model (`LATE-ARCH-RG-08`) then trained separate `R1` and `R2+` logistic paths and routed them at validation/inference time, scoring `flat_brier = 0.131950` on the 2023–2024 held-out window and then passing the 2025 sanity check at `0.106055`. That cleared the routed women model to replace `ARCH-04B` as the active women leader. The tuned/carryover Elo replacement beat raw `ARCH-04`, and its 2025 holdout sanity check (`f6eac553495949a5805c84fb4cdef757`) also passed cleanly. Benchmark-guided challenger design (`LATE-EXT-01/02`) is now also complete: comparing the frozen leaders against `data/reference` on the same all-pairs universe identified the highest recurring women gap in `R2+ / very_likely` games (`played_games = 323`, `play_prob_mass = 321.6`, `mean_brier_gap = 0.021748`) and a second women gap in `R1 / definite` (`played_games = 432`, `mean_brier_gap = 0.007529`), while men gaps were smaller and later-round concentrated in `R2+ / likely` and `R2+ / plausible`. Those benchmark cells are now recorded as challenger-design inputs only; they do not change model selection, which remains anchored to played-game held-out flat Brier. The market-data audit (`LATE-VAL-06`) is now also complete: men BetExplorer coverage is operationally usable from `2021` onward, with regular-season coverage above `97%` and tournament coverage at `100%` for `2021-2025`, while women coverage is effectively unusable historically (`0%` through `2024`, then only `25.3%` regular-season and `85.1%` tournament coverage in `2025`). On the covered tournament subset, market-only probabilities slightly trail the frozen men leader overall (`0.186486` vs `0.184178` across `940` games) but beat it in a few seasons, so `LATE-MKT-01` is viable as a men-only challenger. Women market challengers remain blocked unless coverage is backfilled or a stronger women market source is added. The first men `LATE-MKT-01` slice then added a regular-season BetExplorer market-implied strength feature to the frozen men reference path, but it scored `flat_brier = 0.197276` and `log_loss = 0.579413` on the 2023–2024 held-out window (`2023 = 0.207453`, `2024 = 0.187099`), so it did not beat the frozen men leader `0.195566`. The first `LATE-RATE-02` slice then added a leakage-safe ESPN women four-factor strength differential to the routed women model, and it improved the 2023–2024 held-out window to `flat_brier = 0.130427` with `log_loss = 0.400684`. The 2025 sanity check also passed at `flat_brier = 0.103438` and `log_loss = 0.328220`, which is better than the prior routed women sanity check. That clears the women ESPN four-factor challenger to replace `LATE-ARCH-RG-08` as the active women leader. The follow-up `LATE-EXT-03` slice then exposed ESPN four-factor components individually, instead of using only the composite strength feature, on top of the current men and women leader families. That richer decomposition failed in both leagues: men scored `flat_brier = 0.199281`, and women scored `0.136654`, so it does not advance. The follow-up `LATE-ARCH-DW-01` slice then applied exponential season-recency decay weighting (`decay_base = 0.9`) to the current men and women leader families. It also lost in both leagues: men scored `flat_brier = 0.199205`, and women scored `0.132470`, which is closer but still behind the current women leader `0.130427`. The follow-up `LATE-ARCH-META-01` slice then regenerated OOF base predictions and fit a logit-Ridge stack over the strongest available near-miss members. That also failed decisively: the women stack scored `flat_brier = 0.135684`, and the men stack was much worse at `0.205593`, so the current member set is too correlated or too weak for this meta-learner to help. The follow-up bundled late-feature slice (`LATE-FEAT-24/27/28/29/31`) then added late-5 split form, site profiles, win-quality bins, conference percentile rank, and tournament pedigree on top of the current leader families. That also failed in both leagues: men scored `flat_brier = 0.200470`, and women scored `0.133791`, so the bundled contextual features do not advance. Gate 3 submission plumbing is now implemented, and `VAL-03` is also complete on the historical dry-run path: the combined frozen submission builder succeeded end to end on a historical 2025 combined sample, and bracket diagnostics did not reveal an obvious high-likelihood bucket failure for either league. The remaining live blocker is data availability: the current raw snapshot does not yet include 2026 regular-season inputs, so the real Stage 2 run must wait for that refresh. The current frozen leaders are the generalization-tuned men reference margin model for men and the routed women + ESPN four-factor model unless a later challenger beats them.

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
