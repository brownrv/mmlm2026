# mmlm2026
mmlm2026 is a clean-room Kaggle research repository for March Machine Learning Mania 2026. Its purpose is to maximize leaderboard performance while maintaining strong experimental discipline, reproducibility, and long-term learning.

## Competition references

- Dataset: `data/raw/march-machine-learning-mania-2026/`
- Overview/rules source: `docs/march-machine-learning-mania-2026 - Overview and Data and Rules.docx`
- Dataset relationships: `docs/data/RELATIONSHIP_DIAGRAM.md`
- Tournament round assignment: `docs/data/TOURNEY_ROUND_ASSIGNMENT.md`
- Team spellings policy: `docs/data/TEAM_SPELLINGS_POLICY.md`

Key rules to keep front-of-mind:
- Combined men's + women's forecasting in one submission.
- Brier score evaluation.
- Submission `ID` is `Season_LowTeamID_HighTeamID`.
- `Pred` is probability that the lower TeamID team wins.
- NCAA tournament round is determined from normalized seed pairs, not `DayNum`.
- Team name mapping should use `data/TeamSpellings.csv` as canonical.

## Dev commands

```powershell
uv sync
uv run pre-commit install
uv run pre-commit run --all-files
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src
```

Pre-commit workflow:
1. Run `uv run pre-commit install` once per clone.
2. Run `uv run pre-commit run --all-files` before opening a PR.
3. Let the git hook run on every commit and stage any auto-fixes it applies.

PowerShell helper:

```powershell
./scripts/dev.ps1 sync
./scripts/dev.ps1 hooks-install
./scripts/dev.ps1 hooks-run
./scripts/dev.ps1 test
./scripts/dev.ps1 lint
./scripts/dev.ps1 format
./scripts/dev.ps1 typecheck
./scripts/dev.ps1 all
```
