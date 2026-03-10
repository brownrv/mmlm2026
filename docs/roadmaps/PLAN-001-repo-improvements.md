# PLAN-001: Repository Improvement and Agent UX Hardening

**Created:** 2026-03-09
**Source:** Full repository review — Claude Code session feedback
**Status tracking:** `docs/roadmaps/EXECUTION_STATUS.md`
**Archive when complete:** Move to `docs/roadmaps/archive/PLAN-001-repo-improvements.md` and update `EXECUTION_STATUS.md`

---

## Purpose

This plan addresses gaps identified during a full repository review on 2026-03-09.
Issues span three categories: critical safety/correctness problems, documentation hygiene,
and improvements to AI agent workflow quality. Phases are ordered so that Phase 1 can be
executed immediately without any human decisions.

## Conventions

- **[HUMAN DECISION REQUIRED]** — Agent should create the structure and clearly mark
  placeholders. The task is not complete until a human fills in the decision.
- **Acceptance criteria** — The specific verifiable condition an agent should check before
  marking a task done.
- Tasks within a phase are independent unless a dependency is noted.
- All changes to `AGENTS.md` must include a matching change to `CLAUDE.md` in the same
  commit (sync policy). The CI script `scripts/check_changed_file_policies.py` enforces this.
- If a task touches `src/mmlm2026/`, the memory discipline pre-commit hook will fire.
  Either update `docs/experiments/` or `docs/decisions/` in the same commit, or use
  `ALLOW_NO_MEMORY_UPDATE=1` if the change is purely documentary (e.g., adding a docstring).

---

## Phase 1 — Critical Blockers

These three tasks address active risks. Execute and commit before any other work.

---

### P1-1: Add `mlruns/` to `.gitignore`

**Why:** MLflow stores experiment tracking data in `./mlruns/` by default. This directory
is not currently gitignored. Without this fix, running any experiment will produce files
that could be accidentally committed — potentially including large parquet or artifact files
that are very difficult to remove from git history.

**Files to change:** `.gitignore`

**Instructions:**
In the `# Project-specific` section at the bottom of `.gitignore`, add `mlruns/` after
the `logs/` entry:

```
# Project-specific
.venv/
data/*
logs/
mlruns/
!docs/data/
...
```

**Acceptance criteria:** `.gitignore` contains `mlruns/` in the project-specific section.
Running `git check-ignore -v mlruns/` confirms the directory is ignored.

**Status:** [x] Complete

---

### P1-2: Add AI session restart reminder to `MASTER_CHECKLIST.md`

**Why:** Claude Code and similar agents load `CLAUDE.md` once at session start. If `CLAUDE.md`
or `AGENTS.md` is updated during a project, the running agent continues operating from the
old policy until the session is restarted. This caused a live submission format discrepancy
in the current session (the old `Season_Team1_Team2` format remained loaded despite the file
being corrected to `Season_LowTeamID_HighTeamID`). Without an explicit reminder this will
recur every time policy files are updated.

**Files to change:** `MASTER_CHECKLIST.md`

**Instructions:**
In the `## After Reboot / New Session` section, add the following line after
`- [ ] Run \`uv run pre-commit install\` (safe to run repeatedly).`:

```markdown
- [ ] If `AGENTS.md` or `CLAUDE.md` were updated since the last session, restart the
  AI agent session before continuing — agents load these files once at session start
  and will not pick up changes mid-session.
```

**Acceptance criteria:** `MASTER_CHECKLIST.md` contains an explicit restart reminder in
the "After Reboot / New Session" section.

**Status:** [x] Complete

---

### P1-3: Document stdlib-only constraint on the CI pre-uv policy check

**Why:** In `.github/workflows/ci.yml`, `scripts/check_changed_file_policies.py` runs
before `astral-sh/setup-uv`. This works because the script uses only Python stdlib. But
there is no comment explaining this constraint. A future developer adding a non-stdlib
import to that script would break CI in a non-obvious way (the step runs on the runner's
system Python, not the project's managed Python).

**Files to change:** `.github/workflows/ci.yml`

**Instructions:**
Change the step name from:

```yaml
- name: Changed-file policy checks
```

to:

```yaml
- name: Changed-file policy checks (stdlib-only — intentionally runs before uv setup)
```

**Acceptance criteria:** The CI step name contains a comment making the stdlib-only
constraint explicit.

**Status:** [x] Complete

---

## Phase 2 — Documentation Hygiene

These seven tasks clean up inconsistencies and fill gaps that cause agent confusion or
incorrect behavior. Execute after Phase 1 is complete.

---

### P2-1: Consolidate duplicate roadmap files

**Why:** Two roadmap files exist with diverged content:
- `docs/roadmap.md` — has operational priorities (Current Priorities, Next Up, Parking Lot)
- `docs/roadmaps/roadmap.md` — has phased milestones (Near-Term, Mid-Term, Revisit Queue)

`CLAUDE.md`'s Repository Map points to `docs/roadmaps/` as the roadmap location. Agents
reading CLAUDE.md will look there, but find incomplete content. The two files need to be
merged so there is one canonical roadmap.

**Files to change:** `docs/roadmaps/roadmap.md`, `docs/roadmap.md`, `CLAUDE.md`

**Instructions:**
1. Read both `docs/roadmap.md` and `docs/roadmaps/roadmap.md`.
2. Merge their content into `docs/roadmaps/roadmap.md`. Keep both structural approaches:
   retain the milestone phases (Near-Term/Mid-Term) from the existing roadmaps version
   and incorporate the specific priority items (Current Priorities, Next Up, Parking Lot)
   from `docs/roadmap.md` under the appropriate phases. Remove duplicates.
3. Replace the full content of `docs/roadmap.md` with a redirect note only:
   ```markdown
   # Roadmap

   This file is superseded. The canonical roadmap is `docs/roadmaps/roadmap.md`.
   ```
4. In `CLAUDE.md`, in the `## Repository Map` section, update the roadmap entry to remove
   the parenthetical `(docs/roadmap.md is a placeholder starter)` and replace it with
   `docs/roadmaps/roadmap.md is the canonical working roadmap`.

**Acceptance criteria:**
- `docs/roadmaps/roadmap.md` contains the merged content of both former files.
- `docs/roadmap.md` contains only the redirect note.
- `CLAUDE.md` references `docs/roadmaps/roadmap.md` as canonical with no mention of
  `docs/roadmap.md` being a starter.

**Note:** Changing `CLAUDE.md` requires a matching `AGENTS.md` change in the same commit
(sync policy). In this case, if `AGENTS.md` does not need a content change, add a
comment-style note to the `## Repository Structure` tree in `AGENTS.md` making
`docs/roadmaps/roadmap.md` explicit as the canonical roadmap file.

**Status:** [x] Complete

---

### P2-2: Add `league` to recommended MLflow tags in `AGENTS.md` and `CLAUDE.md`

**Why:** This is a combined men's + women's competition. A `league` MLflow tag
(values: `men`, `women`, `combined`) would be one of the most useful run filters.
It was suggested in the original setup plan but was never added to the canonical
recommended tag list in either agent instructions file. Without it, agents will omit
this tag and runs will be unsortable by league.

**Files to change:** `AGENTS.md`, `CLAUDE.md`

**Instructions:**
- In `AGENTS.md`, find the `Recommended tags:` list under `## Experiment Tracking`
  and add `- \`league\`` to the list.
- In `CLAUDE.md`, find the `Recommended MLflow tags:` line under
  `## Experiment and Decision Memory` and add `league` to the list.

**Acceptance criteria:** Both `AGENTS.md` and `CLAUDE.md` list `league` as a recommended
MLflow tag. The CI changed-file policy check passes (both files updated together).

**Status:** [x] Complete

---

### P2-3: Add MLflow UI access and backup guidance to `AGENTS.md`

**Why:** There is no documentation on where MLflow stores tracking data, how to access
the UI, or how to back it up. Losing MLflow run history near a competition deadline is
painful and hard to reconstruct. Agents writing experiment code also have no reference
for how to launch or check the tracking server.

**Files to change:** `AGENTS.md`, `CLAUDE.md`

**Instructions:**
In `AGENTS.md`, add a new `## MLflow Operations` section immediately after
`## Experiment Tracking`. Add a shorter parallel note in `CLAUDE.md` under
`## Experiment and Decision Memory`. Content:

```markdown
## MLflow Operations

Default tracking URI: `./mlruns/` (relative to project root; gitignored).

Launch the MLflow UI:
```bash
uv run mlflow ui
```
Then open http://localhost:5000 in a browser to browse runs, compare metrics, and
download artifacts.

Backup: before the final competition submission, snapshot `mlruns/` to a separate
backup location (external drive, cloud storage, or a private branch). Do not commit
`mlruns/` to the repository.
```

**Acceptance criteria:** `AGENTS.md` contains an `## MLflow Operations` section with
the tracking URI, UI launch command, and backup recommendation.

**Status:** [x] Complete

---

### P2-4: Mark completed follow-up actions in ADR 0001

**Why:** `docs/decisions/0001-experiment-ledger-is-mlflow.md` has two follow-up actions
marked as unchecked despite both being implemented. Stale unchecked items in decision
records cause agents to re-implement already-completed work.

- "Add git SHA and branch tags to all runs" — done: `start_tracked_run()` in
  `src/mmlm2026/utils/mlflow_tracking.py` auto-attaches `git_sha` and `git_branch`.
- "Create a reusable MLflow logging helper" — done: `src/mmlm2026/utils/mlflow_tracking.py`
  exists and is the canonical helper.

**Files to change:** `docs/decisions/0001-experiment-ledger-is-mlflow.md`

**Instructions:**
In the `## Follow-up Actions` section, change both `- [ ]` items to `- [x]` and append
the implementation reference on the same line:

```markdown
## Follow-up Actions
- [x] Add git SHA and branch tags to all runs — implemented via `git_context()` in
  `src/mmlm2026/utils/mlflow_tracking.py`
- [x] Create a reusable MLflow logging helper — `src/mmlm2026/utils/mlflow_tracking.py`
```

**Acceptance criteria:** Both follow-up items in ADR 0001 are marked `[x]` with
implementation references.

**Status:** [x] Complete

---

### P2-5: Add audience headers to `CLAUDE.md` and `AGENTS.md`

**Why:** Both files coexist in the same repo. Codex agents searching for context will
encounter `CLAUDE.md`. Claude Code agents see both. Without audience labeling,
agent-specific rules can cross-pollinate, causing inconsistent behavior. For example,
Claude Code-specific session restart instructions (P1-2) are irrelevant to Codex.

**Files to change:** `CLAUDE.md`, `AGENTS.md`

**Instructions:**
- At the very top of `CLAUDE.md`, before the `# CLAUDE.md` heading, add:
  ```markdown
  <!-- Audience: Claude Code. Codex and other agents should follow AGENTS.md. -->
  ```
- At the very top of `AGENTS.md`, before the `# AGENTS.md` heading, add:
  ```markdown
  <!-- Audience: Codex and general agents. Claude Code additionally reads CLAUDE.md. -->
  ```

**Acceptance criteria:** Both files have audience HTML comments as their first line.
The CI changed-file policy check passes (both files updated together).

**Status:** [x] Complete

---

### P2-6: Add initial experiment log stub entry

**Why:** `docs/experiments/experiment-log.md` currently contains only the template —
no real entries. Agents checking the log for prior work find nothing and have no
context for what has been tried. With the submission deadline ten days away, an explicit
baseline-state entry is important for orientation.

**Files to change:** `docs/experiments/experiment-log.md`

**Instructions:**
After the template block (after the closing `---`), append a real entry:

```markdown
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
```

**Acceptance criteria:** `docs/experiments/experiment-log.md` contains at least one
real dated entry below the template.

**Status:** [x] Complete

---

### P2-7: Create M+W modeling strategy ADR skeleton

**[HUMAN DECISION REQUIRED]**

**Why:** The competition requires a single combined submission covering both men's and
women's tournaments. None of the agent files address whether to use separate models,
shared feature pipelines, or how to handle calibration differences between M and W.
Without an ADR, agents will make inconsistent choices across sessions. The decision
itself requires human judgment — this task only creates the structure.

**Files to change:** New ADR file in `docs/decisions/` (generated by script)

**Instructions:**
1. Run:
   ```bash
   uv run python scripts/new_decision.py "men-women tournament modeling strategy" --owners "repo maintainers"
   ```
2. Open the generated file (e.g. `docs/decisions/0004-men-women-tournament-modeling-strategy.md`).
3. Fill the `## Context` section with:
   ```
   The competition produces a single submission covering both men's and women's NCAA
   tournaments. A consistent policy is needed for how models are structured, trained,
   and combined for M vs W data to avoid ad-hoc per-session decisions.
   ```
4. Replace the `## Decision` section body with a clearly marked placeholder:
   ```
   **[HUMAN DECISION REQUIRED]** Choose one of the following options and fill in below.

   - Option A: Separate models per league, merged at submission time.
   - Option B: Single shared model with `league` as an explicit categorical feature.
   - Option C: Separate feature pipelines feeding a shared model architecture.
   - Option D: Other (describe).

   Chosen option: ___

   Rationale: ___
   ```
5. In `## Dependencies`, add: `league:men`, `league:women`
6. In `## Invalidated by`, add: `"If competition splits into separate M/W leaderboards"`
7. Add `league` to any MLflow tags used for runs under this strategy.

**Acceptance criteria:** An ADR file exists in `docs/decisions/`, contains the context
paragraph, the decision option list with a blank for the chosen option, and is clearly
marked `[HUMAN DECISION REQUIRED]` in the Decision section.

**Status:** [x] Complete (skeleton created; human decision still required)

---

## Phase 3 — Agent UX Improvements

These five tasks reduce friction and improve the quality of AI agent collaboration.
Execute after Phases 1 and 2 are complete.

---

### P3-1: Narrow memory discipline hook scope in `check_memory_update.py`

**Why:** The hook currently fires when any `src/` path changes without a docs update.
This includes `tests/`, formatting-only commits, and type annotation updates — none of
which warrant experiment or decision notes. Excessive false positives drive bypass
(`ALLOW_NO_MEMORY_UPDATE=1`) habituation, which defeats the purpose of the guard.

**Files to change:** `scripts/check_memory_update.py`

**Instructions:**
Change the `src_changed` line from:
```python
src_changed = any(path.startswith("src/") for path in changed)
```
to:
```python
src_changed = any(path.startswith("src/mmlm2026/") for path in changed)
```
Also update the error message body to reference `src/mmlm2026/` instead of `src/`:
```python
"- Detected staged changes in src/mmlm2026/\n"
```

**Acceptance criteria:** The hook only fires when `src/mmlm2026/` (not `tests/` or
other `src/` paths) is changed without docs updates. Verified by staging a change to
a test file only and confirming the hook passes.

**Note:** This change is in `scripts/`, not `src/mmlm2026/`, so the hook itself will
not trigger on this edit.

**Status:** [x] Complete

---

### P3-2: Add `ruff format --check` to CI

**Why:** CI enforces linting (`ruff check`) but not formatting (`ruff format`).
Pre-commit runs the formatter, but pre-commit can be skipped. Agents submitting PRs
without pre-commit installed will not get formatting feedback until after push. Adding
a check to CI closes this gap.

**Files to change:** `.github/workflows/ci.yml`

**Instructions:**
After the `Ruff lint` step, add a new step:
```yaml
      - name: Ruff format check
        run: uv run ruff format --check .
```

**Acceptance criteria:** CI workflow contains a `Ruff format check` step running
`ruff format --check .` immediately after the lint step.

**Status:** [x] Complete

---

### P3-3: Enable `disallow_untyped_defs` in mypy configuration

**Why:** `AGENTS.md` and `CLAUDE.md` both state that type hints should be used. But
`pyproject.toml` has `disallow_untyped_defs = false`, so mypy will not enforce this.
Agents writing new code will not receive mypy failures for missing return types or
parameter annotations, making the stated policy effectively unenforceable.

**Files to change:** `pyproject.toml`, then fix any mypy errors in `src/mmlm2026/`

**Instructions:**
1. In `pyproject.toml`, under `[tool.mypy]`, change:
   ```toml
   disallow_untyped_defs = false
   ```
   to:
   ```toml
   disallow_untyped_defs = true
   ```
2. Run `uv run mypy src` and fix all type annotation errors that surface in
   `src/mmlm2026/`. Do not suppress errors with `# type: ignore` unless there is a
   documented reason.
3. Confirm `uv run mypy src` passes cleanly before committing.

**Acceptance criteria:** `pyproject.toml` has `disallow_untyped_defs = true` and
`uv run mypy src` passes with zero errors.

**Status:** [x] Complete

---

### P3-4: Expand `HUMAN_AGENT_WORKFLOW.md` with delegation guidance and session template

**Why:** The current file is 20 lines and largely duplicates `AGENTS.md`. It is the
document most likely to improve day-to-day human-agent collaboration quality, but it
does not address: when to delegate to agents vs. act manually, how to verify agent
output before committing, or how to orient an agent at the start of a focused work
session. A session-start template in particular reduces the agent's ramp-up time and
prevents re-reading all policy docs on every session.

**Files to change:** `docs/roadmaps/HUMAN_AGENT_WORKFLOW.md`

**Instructions:**
Expand the file to include all five sections below. Keep the existing
"Standard Loop" and "Guardrails" content; add the three new sections around them.

```markdown
# Human + Agent Workflow

## Standard Loop

1. Create or link an Issue.
2. Write a concrete hypothesis.
3. Explore quickly in notebooks.
4. Move reusable logic to `src/`.
5. Add or update tests.
6. Run and log experiment in MLflow.
7. Update `docs/experiments/`.
8. Update `docs/decisions/` if beliefs change.
9. Add/refresh retest triggers.
10. Commit in small, reviewable slices.

## When to Delegate to an Agent

Good candidates for agent delegation:
- Creating ADR skeletons or experiment note stubs from templates.
- Implementing a `src/` function from a written spec or docstring.
- Writing tests for a specified function.
- Updating documentation to match already-decided code behavior.
- Running lint, format, type, and test checks and fixing mechanical failures.
- Generating a submission file from an existing validated model.

Keep these for human judgment:
- Choosing between modeling strategies (e.g., Option A vs. B in an ADR).
- Deciding whether an experiment result is meaningful or a fluke.
- Selecting the final submission on Kaggle.
- Evaluating whether a leaderboard change justifies a strategy shift.
- Writing the hypothesis for a new experiment family.

## How to Verify Agent Output

Before accepting an agent's changes:
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass.
- [ ] `uv run mypy src` passes.
- [ ] `uv run pytest` passes.
- [ ] `uv run pre-commit run --all-files` passes (or bypass is explicitly documented).
- [ ] Relevant experiment or decision docs were updated if `src/mmlm2026/` changed.
- [ ] Assumption changes are captured (not left only in chat history).
- [ ] No new `# type: ignore` or `ALLOW_NO_MEMORY_UPDATE=1` without a documented reason.

## Session-Start Context Template

Use this prompt to orient an agent at the start of a focused work session.
Copy, fill in the blanks, and paste as your first message.

```
Focus: Issue #___ — [one-line title]
Branch: [branch name or "main"]
Current state: [one sentence from EXECUTION_STATUS.md or roadmap]
Constraints: [any live exceptions to normal rules, or "none"]
Success criteria: [one sentence — what does done look like]
Key files: CLAUDE.md, MASTER_CHECKLIST.md, [any task-specific files]
```

## Guardrails

- If assumptions change, search `docs/experiments/` and `docs/decisions/` for
  impacted work before starting new experiments.
- If an idea is promising but incomplete, open an Issue so it is not lost in chat.
```

**Acceptance criteria:** `HUMAN_AGENT_WORKFLOW.md` contains all five sections:
Standard Loop, When to Delegate, How to Verify Agent Output, Session-Start Context
Template, and Guardrails.

**Status:** [x] Complete

---

### P3-5: Add usage example with recommended tags to `mlflow_tracking.py`

**Why:** `src/mmlm2026/utils/mlflow_tracking.py` documents git context tags but gives
no example of the full recommended tag set from `AGENTS.md`. After P2-2 adds `league`
to the recommended tags, a concrete usage example in the helper reinforces the pattern
for agents writing new experiment code. Without an example, agents will forget tags or
use inconsistent names.

**Files to change:** `src/mmlm2026/utils/mlflow_tracking.py`

**Instructions:**
Expand the `start_tracked_run()` docstring to include a usage example showing all
recommended tags:

```python
def start_tracked_run(
    run_name: str,
    tags: dict[str, Any] | None = None,
) -> mlflow.ActiveRun:
    """Start an MLflow run and attach standard git context tags.

    Recommended tags to pass (see AGENTS.md for full list):
        - hypothesis: one-sentence description of what you expect to learn
        - model_family: e.g. "logistic_regression", "lgbm", "ensemble"
        - league: "men", "women", or "combined"
        - season_window: e.g. "2010-2025"
        - depends_on: e.g. "data:processed_v1", "feature:seed_diff_v2"
        - retest_if: condition that would invalidate this experiment

    Example::

        with start_tracked_run(
            "seed-diff-baseline-v1",
            tags={
                "hypothesis": "seed difference alone predicts win probability well",
                "model_family": "logistic_regression",
                "league": "men",
                "season_window": "2010-2025",
                "depends_on": "data:processed_v1",
                "retest_if": "seed assignment rules change",
            },
        ):
            mlflow.log_params({"feature": "seed_diff", "regularization": "l2"})
            mlflow.log_metrics({"brier_score": 0.21})
    """
```

**Acceptance criteria:** `start_tracked_run()` docstring contains a complete usage
example showing all recommended tags including `league`.

**Note:** This change touches `src/mmlm2026/`, so the memory discipline pre-commit
hook will fire. Use `ALLOW_NO_MEMORY_UPDATE=1` for this commit — it is a docstring-only
change that does not represent a new experiment or modeling decision.

**Status:** [x] Complete

---

## Completion and Archive

When all tasks above are marked `[x]`:

1. Update `docs/roadmaps/EXECUTION_STATUS.md` to mark PLAN-001 as Complete.
2. Move this file to `docs/roadmaps/archive/PLAN-001-repo-improvements.md`.
3. Update `EXECUTION_STATUS.md` to note the archive location.
