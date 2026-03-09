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

## Guardrails

- If assumptions change, search `docs/experiments/` and `docs/decisions/` for impacted work.
- If an idea is promising but incomplete, open an Issue so it is not lost.
