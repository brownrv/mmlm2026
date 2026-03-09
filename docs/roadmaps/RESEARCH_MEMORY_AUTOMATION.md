# Research Memory Automation

This file summarizes what is automated and what still requires human judgment.

## Automated

- `scripts/new_experiment.py`
  - Creates a dated experiment note from template.
  - Appends a stub entry to `docs/experiments/experiment-log.md`.
  - Optional `--create-issue` creates a linked GitHub issue via `gh`.
- `scripts/new_decision.py`
  - Creates next ADR file from template.
  - Enforces required sections: `Dependencies`, `Invalidated by`, `Related Experiments`.
- Pre-commit guard:
  - `scripts/check_memory_update.py` blocks commit when `src/` changes are staged without updates in `docs/experiments/` or `docs/decisions/`.
  - Bypass only for intentional trivial edits: `ALLOW_NO_MEMORY_UPDATE=1`.
- CI policy checks:
  - `scripts/check_changed_file_policies.py` enforces:
    - `AGENTS.md` and `CLAUDE.md` must change together.
    - Kaggle spellings source changes require `data/TeamSpellings.csv` update.

## How to Run

- Create experiment note + log stub:
  - `uv run python scripts/new_experiment.py "<title>" --owner "<owner>"`
- Create ADR from template:
  - `uv run python scripts/new_decision.py "<title>" --owners "<owners>"`
- Run memory-discipline check manually:
  - `uv run python scripts/check_memory_update.py`
- Run changed-file policy checks manually:
  - `uv run python scripts/check_changed_file_policies.py --base <base_sha> --head <head_sha>`

## Still Manual

- Quality and correctness of experiment conclusions.
- Whether a code change truly warrants experiment/decision updates.
- Linking the right MLflow runs/metrics to notes.
