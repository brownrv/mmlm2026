# Retest Triggers

This file tracks broad assumption changes that should cause old work to be reviewed.

## How to use
When a major assumption changes, add an entry with:
- what changed
- which experiments or decisions are likely impacted
- whether retesting is required

---
## YYYY-MM-DD — <assumption or system changed>

Change:
- <what changed>

Why it matters:
- <why previous results might no longer hold>

Impacted items:
- docs/experiments/<file>.md
- docs/decisions/<file>.md
- MLflow runs tagged with `depends_on=<value>`

Required action:
- [ ] Re-run baseline
- [ ] Re-run affected experiment family
- [ ] Update decision record
