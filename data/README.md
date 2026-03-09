# Data Directory

This folder stores local project data artifacts.

- `raw/`: original, immutable source data
- `interim/`: intermediate outputs from cleaning or joins
- `processed/`: model-ready datasets
- `features/`: engineered feature tables
- `submissions/`: generated competition submission files
- `manifests/`: data versioning and metadata manifests

Current Kaggle competition snapshot:
- `raw/march-machine-learning-mania-2026/`

Schema and relationship reference:
- `docs/data/RELATIONSHIP_DIAGRAM.md`

Canonical NCAA round lookup:
- `tourney_round_lookup.csv`
- Round assignment rules: `docs/data/TOURNEY_ROUND_ASSIGNMENT.md`

Canonical team-spellings mapping:
- `TeamSpellings.csv`
- Team spellings policy: `docs/data/TEAM_SPELLINGS_POLICY.md`

`data/` is git-ignored in this repository. Keep large files and sensitive data here, not in versioned code paths.
