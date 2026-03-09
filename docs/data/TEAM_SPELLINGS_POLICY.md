# Team Spellings Policy

Use `data/TeamSpellings.csv` as the canonical/master team-spelling mapping.

## Canonical Source

- `data/TeamSpellings.csv`

This file is intended to be a superset that includes:
- Kaggle spellings
- internal aliases
- external-system linkage fields (for example `espn_id`)

## Kaggle Source Files

- `data/raw/march-machine-learning-mania-2026/MTeamSpellings.csv`
- `data/raw/march-machine-learning-mania-2026/WTeamSpellings.csv`

These are treated as upstream input sources, not the final mapping used by project code.

## Required Behavior

- Prefer `data/TeamSpellings.csv` for all spelling-to-team mapping.
- Do not split logic between Kaggle `MTeamSpellings` and `WTeamSpellings` when master mapping is available.
- When new Kaggle spellings appear, append missing entries to `data/TeamSpellings.csv` so it remains a superset.
