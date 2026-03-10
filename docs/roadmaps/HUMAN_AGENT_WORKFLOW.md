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
- Choosing between modeling strategies (for example Option A vs. B in an ADR).
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

```text
Focus: Issue #___ - [one-line title]
Branch: [branch name or "main"]
Current state: [one sentence from EXECUTION_STATUS.md or roadmap]
Constraints: [any live exceptions to normal rules, or "none"]
Success criteria: [one sentence - what does done look like]
Key files: CLAUDE.md, MASTER_CHECKLIST.md, [any task-specific files]
```

## Guardrails

- If assumptions change, search `docs/experiments/` and `docs/decisions/` for impacted work.
- If an idea is promising but incomplete, open an Issue so it is not lost.
