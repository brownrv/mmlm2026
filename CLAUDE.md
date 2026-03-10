<!-- Audience: Claude Code. Codex and other agents should follow AGENTS.md. -->
# CLAUDE.md

Concise Claude-specific operating guide for this repo.
This file must stay aligned with `AGENTS.md`.

## Sync Policy (No Drift)

- `AGENTS.md` and `CLAUDE.md` are a paired policy set.
- Any rule change in one file must be reflected in the other in the same PR.
- If there is conflict, treat `AGENTS.md` as canonical and immediately update `CLAUDE.md`.
- `MASTER_CHECKLIST.md` is the operational checklist source of truth; update it when workflows/tools/policies change.

## Project Scope

- Kaggle March Machine Learning Mania modeling and data pipelines.
- Goal: reproducible experiments, modular features, MLflow tracking, reliable submissions.
- Stack: Python 3.11.

## Repository Map

- Code: `src/mmlm2026/`
- Notebooks: `notebooks/`
- Scripts: `scripts/`
- Tests: `tests/`
- Data: `data/raw/`, `data/interim/`, `data/processed/`, `data/features/`, `data/submissions/`, `data/manifests/`
- Docs: `docs/decisions/`, `docs/experiments/`, `docs/roadmaps/` (`docs/roadmaps/roadmap.md` is the canonical working roadmap)

## Development Rules

- Keep production logic in `src/`; move reusable notebook logic into `src/`.
- Never leave durable production logic only in notebooks.
- Prefer small functions, explicit naming, type hints, and docstrings for public APIs.
- Use absolute imports (example: `from mmlm2026.features.seed_features import build_seed_features`).
- Avoid large refactors unless requested.
- Write small, reviewable commits.
- Do not silently change evaluation assumptions or evaluation logic.

## Data Rules

- Never modify `data/raw/`.
- Current Kaggle raw snapshot is in `data/raw/march-machine-learning-mania-2026/`.
- Use `data/interim/` for intermediate transforms/joins.
- Use `data/processed/` for cleaned structured datasets.
- Use `data/features/` for model-ready feature tables.
- Use `data/submissions/` for generated submission files.
- Use `data/manifests/` for data/version metadata.
- Use `docs/data/RELATIONSHIP_DIAGRAM.md` as the canonical join-key and table-relationship reference.
- Use `docs/data/TOURNEY_ROUND_ASSIGNMENT.md` as the canonical NCAA round-assignment reference.
- Use `docs/data/TEAM_SPELLINGS_POLICY.md` as the canonical team-spellings policy.
- Determine NCAA round from normalized seed-pair lookup (`data/tourney_round_lookup.csv`), not from `DayNum`.
- `DayNum < 134` filtering for Elo cutoffs is separate and still valid.
- Prefer `data/TeamSpellings.csv` for mapping team name spellings; treat Kaggle `MTeamSpellings.csv` and `WTeamSpellings.csv` as upstream sources only.

## Experiment and Decision Memory

- Track significant runs in MLflow with params, metrics, artifacts, commit SHA, branch, data/feature version, and hypothesis.
- Recommended MLflow tags: `hypothesis`, `depends_on`, `retest_if`, `model_family`, `season_window`, `league`.
- MLflow operations:
- default tracking URI is `./mlruns/` (gitignored)
- launch UI with `uv run mlflow ui` and open `http://localhost:5000`
- before final submission, back up `mlruns/` to a separate location (do not commit it)
- Stable decisions go in `docs/decisions/`.
- Experiment notes, failures, caveats, and retest triggers go in `docs/experiments/`.
- Use markers in notes: `Dependencies:`, `Invalidated by:`, `Re-test if:`.
- After meaningful tests, update experiment logs in `docs/experiments/`.
- Record assumptions and explicit retest triggers in experiment notes and decision records.
- If an assumption changes, search `docs/experiments/` and `docs/decisions/` for impacted work.
- If a new idea is promising but incomplete, create an Issue so it is not lost in chat.

## Submission Rules

- CSV schema:

```csv
ID,Pred
YYYY_1101_1203,0.63
```

- `ID = Season_LowTeamID_HighTeamID` (lower TeamID first).
- `Pred = probability that the lower TeamID team wins`.
- Do not submit exact `0` or `1`; default clipping: `0.025 <= p <= 0.975`.

## Standard Commands

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pre-commit install
uv run pre-commit run --all-files
uv run python scripts/new_experiment.py "<title>" --owner "<owner>"
uv run python scripts/new_decision.py "<title>" --owners "<owners>"
uv run python scripts/check_memory_update.py
uv run python scripts/check_changed_file_policies.py --base <base_sha> --head <head_sha>
```

Pre-commit workflow:
1. Install once per clone.
2. Run all hooks before PR.
3. Let commit hooks run and stage auto-fixes.

## Preferred Workflow

1. Explore ideas in notebooks.
2. Move reusable logic to `src/`.
3. Add tests.
4. Track runs in MLflow.
5. Record findings in `docs/experiments/`.
6. Create submissions only after validation.
7. Promote stable assumptions into `docs/decisions/` with invalidation criteria.
