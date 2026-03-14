# Competition Operations Guide

Operational reference for March Machine Learning Mania 2026.

Primary source: `docs/march-machine-learning-mania-2026 - Overview and Data and Rules.docx`

Frozen models entering Stage 2:
- Men: generalization-tuned reference margin model
- Women: `COOPER-ARCH-01 + COOPER-ARCH-04 v1` routed women model with ESPN four-factor and conference-rank features

## Submission Mechanics

- Combined competition for men's and women's tournaments.
- Submission format:
  - `ID`: `Season_LowTeamID_HighTeamID`
  - `Pred`: probability that lower TeamID team wins
- Predict all possible matchups listed in sample submission files.
- Select one final submission explicitly before deadline.

## Evaluation

- Metric: Brier score.
- Operationally equivalent to MSE on binary outcomes in this context.

## Timeline (2026 Competition)

- Start: February 19, 2026
- Final submission deadline: March 19, 2026 4PM UTC
- Tournament scoring refresh window: March 19 - April 6, 2026
- Selection Sunday / Stage 2 release: approximately March 15, 2026

Always confirm dates on the Kaggle competition page before final submissions.

## Limits and Governance

- Max team size: 5
- Max submissions per day: 5
- Final judging uses one selected submission
- Data usage/license in competition rules: CC-BY 4.0 (per rules summary)

## Data Update Operations

- Kaggle may release updated current-season files near tournament start.
- On each update:
  1. Refresh raw files in `data/raw/march-machine-learning-mania-2026/`
  2. Update `data/manifests/` metadata
  3. Confirm the raw snapshot includes current-season regular-season files for the target submission season
  4. Confirm all external dependencies used by the frozen pair are refreshed as well, especially the women ESPN boxscore ingest needed for the frozen women four-factor feature
  5. Re-run the frozen submission builder and validation checks
  6. Record operational outcome in `docs/experiments/experiment-log.md` if behavior changes materially
- When the official Selection Sunday release lands, prefer the revised `2021-2026` detailed files if they match the supplemental correction set already audited locally; if they differ materially, reopen the targeted refresh gate before freezing the final live candidate.

## Live Commands

### Pre-Selection Sunday

Use the historical dry run to confirm the frozen submission path still works end to end:

```powershell
uv run python scripts/build_frozen_submission.py --sample-submission data/interim/2025_combined_stage2_sample.csv --season 2025 --save-artifacts --tag gate3-dryrun-2025
```

Validate the resulting dry-run submission explicitly:

```powershell
uv run python scripts/validate_submission.py data/submissions/2025_gate3-dryrun-2025_submission.csv --sample data/interim/2025_combined_stage2_sample.csv
```

### Selection Sunday / Stage 2 Live Run

After Kaggle publishes the real 2026 Stage 2 sample, the raw snapshot contains 2026 regular-season inputs, and the 2026 women ESPN ingest is available:

```powershell
uv run python scripts/build_frozen_submission.py --sample-submission data/raw/march-machine-learning-mania-2026/SampleSubmissionStage2.csv --season 2026 --save-artifacts --tag final-candidate
```

Validate the live submission before upload:

```powershell
uv run python scripts/validate_submission.py data/submissions/2026_final-candidate_submission.csv --sample data/raw/march-machine-learning-mania-2026/SampleSubmissionStage2.csv
```

### Optional Final Checks

Run the standard repo verification before upload if code changed after the last clean commit:

```powershell
uv run ruff check .
uv run mypy src
uv run pytest
```

## Expected Outputs

The frozen submission builder writes:
- submission CSV: `data/submissions/<season>_<tag>_submission.csv`
- optional men/women prediction parquet files under `data/submissions/<season>_<tag>_artifacts/`
- optional bracket diagnostics under `data/submissions/<season>_<tag>_artifacts/men_bracket/` and `.../women_bracket/`

Historical dry-run outputs already verified:
- `data/submissions/2025_gate3-dryrun-2025_submission.csv`
- `data/submissions/2025_gate3-dryrun-2025_artifacts/`
- `data/submissions/2025_gate3-dryrun-2025-refresh_submission.csv`
- `data/submissions/2025_gate3-dryrun-2025-refresh_artifacts/`

## Current Live Blocker

- The current raw snapshot does not yet include 2026 regular-season inputs.
- The frozen women model also requires the 2026 ESPN ingest for women four-factor features.
- Because of those data dependencies, the real 2026 Stage 2 sample cannot be scored end to end yet.
- No code or model-selection work remains blocked by this; only the final live data refresh is pending.

## Upload Checklist

- [ ] `SampleSubmissionStage2.csv` is present locally and for season `2026`
- [ ] 2026 regular-season inputs are present in `data/raw/march-machine-learning-mania-2026/`
- [ ] 2026 women ESPN boxscore ingest is present under `data/processed/espn/womens-college-basketball/2026/`
- [ ] Frozen builder command completed successfully
- [ ] `scripts/validate_submission.py` passed on the final candidate
- [ ] Final candidate file exists in `data/submissions/`
- [ ] Output file name and tag are recorded in `docs/experiments/experiment-log.md`
- [ ] Upload the validated file to Kaggle
- [ ] Explicitly select the final submission in the Kaggle UI before the deadline
- [ ] Record the selected file name and Kaggle timestamp in project notes

## Failure Handling

- If the live run fails because 2026 data is missing, stop and refresh the Kaggle raw snapshot first.
- If submission validation fails, do not upload; fix row coverage or ID/schema issues locally and rerun the builder.
- If bracket diagnostics fail but the submission file validates, treat the submission file as primary and investigate the diagnostics separately unless the failure indicates a data-join bug.

## Project-Specific Guardrails

- Do not derive NCAA round from `DayNum`; use seed-pair lookup policy.
- Use `data/TeamSpellings.csv` as canonical name mapping.
- Keep probabilities clipped unless intentionally changed and documented.
- Do not replace frozen models during Stage 2 for subjective reasons, added complexity, or leaderboard curiosity.
