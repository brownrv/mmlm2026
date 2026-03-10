# MASTER_CHECKLIST.md

One-stop operational checklist for this project.

Use this file when you are unsure what to do next.
Keep it updated whenever workflows, automation, or policies change.

## Always-True Rules

- Keep reusable logic in `src/`, not only in notebooks.
- Keep `AGENTS.md` and `CLAUDE.md` aligned.
- Keep `data/TeamSpellings.csv` as canonical team-name mapping.
- Do not infer NCAA tournament round from `DayNum`; use seed-pair lookup policy.
- Do not modify `data/raw/`.

## After Reboot / New Session

- [ ] Open this file first: `MASTER_CHECKLIST.md`.
- [ ] Run `uv sync`.
- [ ] Run `uv run pre-commit install` (safe to run repeatedly).
- [ ] If `AGENTS.md` or `CLAUDE.md` were updated since the last session, restart the
  AI agent session before continuing; agents load these files once at session start and
  will not pick up changes mid-session.
- [ ] Run `uv run pre-commit run --all-files`.
- [ ] Confirm branch and status: `git branch --show-current`, `git status`.
- [ ] Review current priorities:
- `docs/roadmaps/roadmap.md`
- `docs/roadmaps/EXECUTION_STATUS.md`
- [ ] Check open issues for active work and blockers.

## Before Starting Work

- [ ] Confirm task has an Issue, or create one if missing.
- [ ] Write or update a hypothesis for the work.
- [ ] Identify impacted assumptions and linked decisions/experiments.
- [ ] If assumptions changed, scan:
- `docs/experiments/`
- `docs/decisions/`

## When You Have a New Idea

- [ ] Create a GitHub Issue using `.github/ISSUE_TEMPLATE/new_idea.md`.
- [ ] Fill: summary, expected benefit, why it might work, dependencies, risks, test plan.
- [ ] Link idea to relevant experiment notes or decisions.
- [ ] If not implementing now, move issue to backlog/board explicitly.

## How to Start a New Experiment

- [ ] Generate experiment note:
- `uv run python scripts/new_experiment.py "<title>" --owner "<owner>"`
- [ ] Optional: create linked issue with `--create-issue`.
- [ ] Confirm experiment stub was appended to `docs/experiments/experiment-log.md`.
- [ ] Define success criteria and baseline before coding.
- [ ] Tag MLflow run with at least:
- `hypothesis`, `depends_on`, `retest_if`, `model_family`, `season_window`

## How to Start a New Decision Record

- [ ] Generate ADR:
- `uv run python scripts/new_decision.py "<title>" --owners "<owners>"`
- [ ] Fill required sections:
- `Dependencies`, `Invalidated by`, `Related Experiments`
- [ ] Link to experiment notes and MLflow runs.

## During Implementation

- [ ] Keep commits small and reviewable.
- [ ] Update docs when behavior/policy changes, not afterward.
- [ ] If code changes assumptions, update experiment/decision memory in same work cycle.
- [ ] If touching `AGENTS.md`, update `CLAUDE.md` in the same commit.

## Data and Mapping Hygiene

- [ ] Use canonical references:
- team names: `data/TeamSpellings.csv`
- round assignment: `data/tourney_round_lookup.csv` + `src/mmlm2026/round_utils.py`
- [ ] If Kaggle `MTeamSpellings.csv` or `WTeamSpellings.csv` changes:
- update `data/TeamSpellings.csv` to remain a superset.
- [ ] If Kaggle raw files refresh:
- update `data/manifests/`
- rerun baseline checks and document changes.

## Before Commit

- [ ] Run format/lint/type/test:
- `uv run ruff format .`
- `uv run ruff check . --fix`
- `uv run mypy src`
- `uv run pytest`
- [ ] Run full hooks:
- `uv run pre-commit run --all-files`
- [ ] Confirm memory discipline:
- `uv run python scripts/check_memory_update.py`
- [ ] Ensure docs were updated if `src/` changed.
- [ ] Ensure issue links and experiment/decision links are present.

## Before Push / PR

- [ ] Recheck branch and diff scope.
- [ ] Ensure commit message explains intent and impact.
- [ ] Ensure assumptions/retest triggers are captured in docs.
- [ ] Ensure CI policy constraints will pass:
- AGENTS/CLAUDE changed together when applicable.
- canonical TeamSpellings updated when Kaggle spelling sources changed.
- [ ] Push and watch CI.

## After Merge

- [ ] Update roadmap/progress docs if milestone state changed.
- [ ] Close or update linked issues.
- [ ] Add follow-up issues for deferred work.
- [ ] Record any new retest triggers in `docs/experiments/retest-triggers.md`.

## Competition Ops (When Near Deadline)

- [ ] Confirm latest competition timeline from Kaggle page.
- [ ] Pull latest Kaggle data update if available.
- [ ] Rebuild features and rerun final validation.
- [ ] Validate submission schema:
- `ID = Season_LowTeamID_HighTeamID`
- `Pred = P(lower TeamID wins)`
- [ ] Ensure probabilities are bounded (no exact 0 or 1 unless explicitly intended and justified).
- [ ] Select final submission explicitly on Kaggle.

## Weekly Maintenance

- [ ] Review stale experiments and unresolved `Re-test if` triggers.
- [ ] Review ADRs for assumptions likely invalidated.
- [ ] Reconcile roadmap with current issue backlog.
- [ ] Confirm this checklist still matches actual workflow and tooling.

## Automation Reference

- New experiment note:
- `uv run python scripts/new_experiment.py "<title>" --owner "<owner>"`
- New ADR:
- `uv run python scripts/new_decision.py "<title>" --owners "<owners>"`
- Memory check (local):
- `uv run python scripts/check_memory_update.py`
- Changed-file policy check (CI/local):
- `uv run python scripts/check_changed_file_policies.py --base <base_sha> --head <head_sha>`

## If You Are Unsure

- [ ] Do the smallest safe step that preserves traceability.
- [ ] Open/update an Issue instead of leaving ideas in chat only.
- [ ] Add a brief experiment or decision note rather than relying on memory.
