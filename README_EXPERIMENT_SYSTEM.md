# Starter System for Decisions and Experiments

This starter pack gives you a lightweight operating system for:
- code history
- experiment memory
- decision logging
- retest triggers

## Suggested source of truth by question

- What changed? -> Git
- Did it help? -> MLflow
- Why did we choose this? -> docs/decisions/
- What should be revisited later? -> docs/experiments/ and issue tracker

## Suggested next steps

1. Copy these files into your repo root
2. Put `AGENTS.md` in the repo root
3. Start using `docs/experiments/experiment-log.md`
4. Create a new experiment note for each meaningful idea family
5. Tag MLflow runs with dependencies and retest triggers
6. Create issues when an assumption change affects old experiments
