# ADR 0003: TeamSpellings.csv is canonical team-name mapping

Status: Accepted
Date: 2026-03-10
Owners: repo maintainers

## Context

Kaggle provides separate spelling files for men and women, but project workflows
need a single mapping layer across both leagues and external systems (for example ESPN).

## Decision

Use `data/TeamSpellings.csv` as the canonical spelling-to-team mapping table.

Treat Kaggle sources as upstream inputs:
- `data/raw/march-machine-learning-mania-2026/MTeamSpellings.csv`
- `data/raw/march-machine-learning-mania-2026/WTeamSpellings.csv`

Project rule:
- keep `data/TeamSpellings.csv` as a superset of Kaggle spellings.

## Alternatives Considered

- Use Kaggle `MTeamSpellings.csv` and `WTeamSpellings.csv` directly in code paths.
- Build mapping dynamically each run from Kaggle raw files only.

## Why This Decision

- One canonical mapping avoids split logic and duplicate handling.
- Supports unified men/women workflows.
- Enables stable extension fields (`espn_id`) for future integrations.

## Consequences

### Positive
- Simplified mapping API for feature/data pipelines.
- Easier integration with external datasets.
- Explicit governance for spelling drift.

### Negative
- Requires maintenance when Kaggle raw spellings are updated.
- Needs policy checks to prevent canonical file drift.

## Dependencies

- `data/TeamSpellings.csv`
- `docs/data/TEAM_SPELLINGS_POLICY.md`
- CI changed-file policy checks in `scripts/check_changed_file_policies.py`

## Invalidated by

- Competition format changes that provide an official unified mapping file with external IDs.
- A project decision to replace this mapping with a maintained external reference service.

## Related Experiments

- docs/experiments/experiment-log.md
- docs/data/TEAM_SPELLINGS_POLICY.md

## Follow-up Actions

- [ ] Add periodic parity check report: Kaggle spellings not in canonical master.
- [ ] Add data test to fail when canonical mapping has duplicate `TeamNameSpelling` entries.
