# mmlm2026
mmlm2026 is a clean-room Kaggle research repository for March Machine Learning Mania 2026. Its purpose is to maximize leaderboard performance while maintaining strong experimental discipline, reproducibility, and long-term learning.

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
