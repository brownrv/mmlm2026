# PLAN-002: Competition Attack Plan

Status: Active
Created: 2026-03-10
Final submission deadline: 2026-03-19 16:00 UTC
Days remaining at creation: 9

---

## 0. Meta-Strategy

**Thesis:** Brier score rewards well-calibrated probabilities on played tournament games, with R1 games dominating the final score because they are guaranteed. Every model choice must answer: "Does this make my probabilities more accurate on played games, with particular attention to calibration on Round of 64 matchups?"

**Operating principles:**

1. **Calibration over discrimination.** A model with perfect rank order but overconfident outputs will score worse than a modestly-ranked but well-calibrated one. Never sacrifice calibration for raw accuracy.
2. **Feature pre-registration.** Define features and their expected direction of effect *before* fitting. Post-hoc rationalization inflates apparent signal.
3. **Reproducible baselines first.** No ensemble without at least two stable, independently-understood baselines.
4. **Log everything.** Every MLflow run gets a hypothesis tag. If you cannot state the hypothesis in one sentence, do not run the experiment.
5. **Evaluate locally; submit once.** All model selection and scoring is done offline. A single Stage 2 submission is made after Selection Sunday. No incremental Kaggle submissions are used for feedback.
6. **Separate signal from noise early.** Carry only features that survive leave-seasons-out cross-validation into ensemble candidates.

---

## 1. Competition Framing

### 1.1 Understanding the Metric

**Metric:** Brier score
**Formula:** `BS = (1/N) * Σ (p_i - y_i)²`
where `p_i` is the predicted probability that the lower-seeded TeamID wins, `y_i ∈ {0, 1}`.

**Strictly proper scoring rule.** Brier score is strictly proper: the unique prediction that minimizes expected Brier score for any game is the true probability — not a rounded value, not a pushed value, not a "confident" signal. Any deviation from the true probability in either direction *always* increases expected score. This is not a modeling preference; it is a mathematical guarantee. The practical consequence: post-processing steps that push probabilities away from model output (rounding, confidence boosting, manual overrides) are harmful unless the model output itself is known to be miscalibrated.

**The overconfidence trap — worked example.**

Consider a game where the true win probability for the favored team is 0.70.

*Perfectly calibrated model predicts 0.70:*
```
E[Brier] = 0.70 * (0.70 - 1)² + 0.30 * (0.70 - 0)²
         = 0.70 * 0.09 + 0.30 * 0.49
         = 0.063 + 0.147 = 0.210
```

*Overconfident model predicts 0.90 for the same game:*
```
If correct (70% of the time):  (0.90 - 1)² = 0.01
If wrong   (30% of the time):  (0.90 - 0)² = 0.81
E[Brier] = 0.70 * 0.01 + 0.30 * 0.81 = 0.007 + 0.243 = 0.250
```

The overconfident model scores **0.250** on this game — identical to predicting 0.5 blindly, and **19% worse** than the calibrated model (0.250 vs 0.210). The "confident" prediction gains almost nothing when right and loses catastrophically when wrong.

**Overconfidence in uncertain games is the primary way to lose this competition.** Most matchups in the submission universe involve genuine uncertainty (seeds 5–12 matchups, cross-region comparisons, all women's games). A model that inflates probabilities beyond what the evidence supports will be penalized across thousands of games simultaneously.

**Implications:**

| Behavior | Effect on Brier Score |
|---|---|
| Predict 0.5 for every game | Guaranteed 0.250 per game; floor to beat |
| Perfectly calibrated model | Optimal; cannot be improved by any re-mapping |
| Overconfident (pushed past true probability) | Always worse in expectation — the 0.90 example above shows a 0.040 penalty |
| Underconfident (pushed toward 0.5) | Worse than calibrated, but less catastrophically than overconfidence |
| Clipping at [0.025, 0.975] | Required; prevents infinite penalty at 0/1 for wrong predictions |

**Target benchmark:**
- Naive 0.5 baseline: ~0.250
- Seed-difference logistic: ~0.220 (historical reference)
- Top-quintile Kaggle entries: ~0.175–0.195 (historical reference)

**Note on scoring scope:** Only *actually played* tournament games are scored. Play-in (First Four) games are excluded. The sample submission file contains all *possible* matchups as a convenience, but Kaggle scores only the subset that occurs. This has a practical implication: Round of 64 games matter most because they are guaranteed to be played, while remote hypothetical matchups matter only if the bracket reaches them. Offline evaluation should therefore optimize the true scoring rule first (flat Brier on held-out played games) and use round-aware diagnostics to confirm strength on high-likelihood games. See §7.1.

### 1.2 Submission Format Constraints

```
ID,Pred
2026_1101_1203,0.63
```

- `ID = Season_LowTeamID_HighTeamID` (lower TeamID always first)
- `Pred` = probability that lower TeamID wins
- Clip: `0.025 <= Pred <= 0.975` (project default; document any deviation)
- Must predict **all** pairs in SampleSubmission files (Stage 1 and Stage 2)
- One submission must be explicitly selected as final before deadline
- Stage 2 sample submission will be released after bracket is set (~March 15)

**Watch-for:** If Stage 2 TeamIDs differ from Stage 1 universe, the submission pipeline must handle the mapping without data leakage.

### 1.3 Data Inventory

See `docs/data/FILE_CATALOG.md` for full catalog. Summary of high-value tables:

| Category | Men | Women | Notes |
|---|---|---|---|
| Team identity | `MTeams.csv` | `WTeams.csv` | TeamID is stable across seasons |
| Tournament seeds | `MNCAATourneySeeds.csv` | `WNCAATourneySeeds.csv` | Primary signal source |
| Regular season compact | `MRegularSeasonCompactResults.csv` | `WRegularSeasonCompactResults.csv` | Win/loss, scores |
| Regular season detailed | `MRegularSeasonDetailedResults.csv` | `WRegularSeasonDetailedResults.csv` | Box-score stats |
| Tourney results | `MNCAATourneyCompactResults.csv` | `WNCAATourneyCompactResults.csv` | Historical outcomes |
| Tourney detailed | `MNCAATourneyDetailedResults.csv` | `WNCAATourneyDetailedResults.csv` | Box-score for tourney games |
| Massey ordinals | `MMasseyOrdinals.csv` | — | ~100 ranking systems, men only |
| Coaches | `MTeamCoaches.csv` | — | Experience/continuity signal |
| Conference tourneys | `MConferenceTourneyGames.csv` | `WConferenceTourneyGames.csv` | Late-season form |
| Game cities | `MGameCities.csv` | `WGameCities.csv` | Neutral-site / home-court proxy |
| Secondary tourney | `MSecondaryTourneyCompactResults.csv` | `WSecondaryTourneyCompactResults.csv` | NIT, etc. |

**Data constraints:**

- `data/raw/` is immutable — never modify
- Canonical round assignment: `data/tourney_round_lookup.csv` (seed-pair, not DayNum)
- Canonical team name mapping: `data/TeamSpellings.csv`
- Elo regular-season cutoff: `DayNum < 134` (rule still valid; separate from round assignment)

### 1.4 Feature Pre-Registration

Features must be registered before fitting. For each feature, record:
- Name and derivation formula
- Expected direction (positive/negative effect on lower-TeamID win probability)
- Data dependency (which raw tables)
- Potential leakage risk (Y/N)
- Invalidation condition

**Pre-registered feature families (v1):**

| Feature Family | Expected Direction | Leakage Risk | Notes |
|---|---|---|---|
| Seed difference (LowTeam seed − HighTeam seed) | + lower seed wins | None | Primary baseline |
| Elo rating difference (end of regular season) | + higher Elo wins | None; cutoff at DayNum<134 | Classic signal |
| Win percentage (regular season) | + higher W% wins | None | Simple strength |
| Strength of schedule (opponent avg Elo) | + harder schedule team | None | Context for W% |
| Point differential (avg margin) | + larger margin wins | None | Often > W% alone |
| Tempo (possessions per 40 min proxy) | Interaction | None | Matchup interaction |
| Efficiency (offensive/defensive rating) | + better efficiency | None | From detailed results |
| SOS-adjusted net efficiency | + stronger adjusted team wins | None | `net_eff = off_eff - def_eff`; season-normalize `net_eff` and `sos`, then sum into a combined strength index |
| Massey ordinal consensus rank | + better rank | None | Men only |
| Seed round expectation (historical seed win rates) | + favored seed | None | Round-specific |
| Coach experience (tourney appearances) | + experienced coach | None | Men only |
| Conference strength (avg Elo of conference peers) | + stronger conf | None | Indirect signal |
| Recent form (last N games win rate) | + hot team | None | Define N pre-fit |
| Head-to-head (current season, if any) | + H2H winner | Low | Rare in tourney matchups |

---

## 2. Evaluation Framework

### 2.1 Cross-Validation Design

**Primary CV scheme:** Leave-seasons-out (temporal split)

```
Train seasons: 2010–2022  (floor; see §2.4 for rationale)
Validation seasons: 2023–2024 (held-out tournament outcomes only)
Test (never-touch): 2025 tournament outcomes
Submission target: 2026 tournament
```

**Rationale:** NCAA basketball has structural drift (rule changes, three-point line, conference realignment). Weighting recent seasons more heavily may improve generalization. Train on all available history above the floor but evaluate on recency.

**Evaluation unit:** Tournament matchup pair (LowTeamID, HighTeamID, Season) — played games only, play-in excluded
**Primary CV metric:** Flat Brier on held-out played tournament games (non-mirrored only; see §2.3)
**Secondary metrics:** `R1` Brier, `R2+` Brier, and men/women split Brier for diagnostic focus
**Tertiary metric:** Log-loss (calibration diagnosis)
**Optional diagnostic metric:** Fixed-weight round-aware Brier (see §7.1) for prioritizing high-likelihood games without changing the optimization target
**Do not use:** Regular-season game Brier score as a proxy for tournament Brier score

### 2.2 Temporal Leakage Policy

**No future data in training — ever.** This means no feature, no summary statistic, no normalization constant may be derived from data that post-dates the game being predicted. Violations are subtle and must be audited explicitly:

| Leakage vector | Example | Mitigation |
|---|---|---|
| Future game outcomes used in feature computation | Elo computed over full season including tourney games | Cutoff all Elo updates at `DayNum < 134` |
| Cross-season statistics contaminated by future seasons | "Team X has a 65% historical tourney win rate" computed over all seasons including the target year | Compute historical rates using only seasons strictly before the target season |
| Calibration fitted on test fold | Isotonic regression trained on 2025 validation outcomes | Calibration must be trained on out-of-fold predictions from training seasons only |
| Normalization using global statistics | Standardizing features with mean/std computed over all seasons | Fit scaler on training folds only; apply frozen scaler to validation/test |
| Imputed values derived from full dataset | Filling missing women's Massey rank with full-dataset median | Impute using training-fold statistics only |
| Massey ordinals from after selection | Using a ranking with `RankingDayNum ≥ 134` as a pre-tournament strength feature | Filter Massey to `RankingDayNum ≤ 133`; use the final pre-cutoff ranking for each system |

**Enforcement:** Before any experiment advances to Gate 1, run the leakage audit checklist in §5.2. Log the audit result as an MLflow tag: `leakage_audit: passed`.

### 2.3 Mirror Training and Evaluation

**Mirror augmentation.** Each training matchup `(TeamA, TeamB, outcome=1)` is paired with its mirror `(TeamB, TeamA, outcome=0)` — all directional features negated, label flipped. This:

1. Doubles the effective training size at no additional data cost
2. Forces the model to learn a symmetric probability function: `P(A beats B) = 1 - P(B beats A)`
3. Prevents the model from learning a spurious directional bias introduced by the `LowTeamID-first` submission format

**Mirror construction rules:**
- All difference features (e.g. `elo_diff = elo_A - elo_B`) negate: `elo_diff_mirror = elo_B - elo_A`
- All absolute features (e.g. `elo_A`, `elo_B`) swap: `elo_A_mirror = elo_B`
- Label flips: `outcome_mirror = 1 - outcome`
- Mirror rows are tagged with `is_mirror = True` in the training table for auditability

**Evaluation uses non-mirrored games only.** When computing Brier score on validation or test folds, use only the original (non-mirror) rows. Including both mirrors would double-count every game, artificially halving variance estimates and making scores appear more stable than they are. The submission itself is naturally non-mirrored (one row per pair), so non-mirrored evaluation is the correct analogue.

**Prediction-time orientation.** At submission time, always pass features in the canonical `LowTeamID-first` orientation — never the mirror. For models using only difference features (e.g. `elo_diff = elo_low - elo_high`), the orientation is enforced by the sign convention. For models that also use absolute features (e.g. `low_elo`, `high_elo`), the `Low`/`High` labels must be assigned before prediction and must match the training convention. Do not predict on mirror-orientation inputs and attempt to invert the result.

### 2.4 Minimum Training Size Floor

**Minimum season floor: 2010 (men and women).**

Rationale:
- Detailed results (box-score stats) are available for men starting 2003, women starting 2010. Using pre-2010 data for women requires degrading to compact-only features, creating a training set inconsistency that may hurt calibration more than it helps with volume.
- 2009 and earlier seasons predate significant rule and conference structure changes that make them less predictive of modern outcomes.
- The floor ensures at least one full decade of data before any validation season.

**Minimum game count check:** Before training on any season fold, assert that the training set contains ≥ 50 tournament games (non-mirrored). If a fold falls below this threshold, expand the training window rather than proceeding with an underpowered model. Log the actual game count as an MLflow param: `train_tourney_games`.

| League | Detailed results available from | Season floor |
|---|---|---|
| Men | 2003 | 2010 |
| Women | 2010 | 2010 |

### 2.5 Validation Pipeline

```
scripts/validate_cv.py (to be built)
  - Input: feature table, model config
  - Output: per-season Brier score (non-mirrored), calibration curve, reliability diagram
  - Artifacts: logged to MLflow per run
  - Asserts: train_tourney_games >= 50 per fold; leakage_audit tag present
```

**Reliability diagram checkpoints:**
- 10 calibration bins
- Flag if any bin deviates >0.05 from diagonal
- Isotonic regression or Platt scaling if uncalibrated

### 2.6 Overfitting Guards

- Never tune hyperparameters on the 2025 holdout set
- All feature selection must use only 2010–2022 CV folds
- If 2025 score is >0.010 worse than CV score, investigate before submitting
- Normalization and calibration parameters frozen from training folds before applying to validation/test

---

## 3. Core Feature Engineering

### 3.1 Build Order (Dependency-Ordered)

```
Phase A — Foundation (must exist before any model)
  A1. Team season summary (regular season W%, avg margin, avg score)
  A2. Elo ratings (end-of-regular-season, DayNum < 134 cutoff)
  A3. Seed lookup table (Season, TeamID -> Seed)

Phase B — Enhanced Strength
  B1. Adjusted efficiency (off/def rating from detailed results)
  B2. Strength of schedule (opponent avg Elo)
  B3. Recent form (last 10 games W%)
  B4. Massey ordinal consensus (men only; median rank across systems)
  B5. SOS-adjusted net efficiency (season-normalized net_eff + sos)

Phase C — Matchup Features
  C1. Elo difference (per matchup pair)
  C2. Seed difference (per matchup pair)
  C3. Efficiency matchup (off_A - def_B, off_B - def_A)
  C4. Tempo interaction (fast vs slow team pairing)
  C5. Historical seed-pair outcome rate (base rate by seed pair)

Phase D — Contextual
  D1. Coach experience (men only)
  D2. Conference strength (avg conference Elo)
  D3. Geographic/neutral-site flag
  D4. Round-specific priors (historical win rates by seed in each round)
```

### 3.2 Feature Table Schemas

All feature tables stored in `data/features/` as Parquet.

**Matchup feature table schema (canonical):**

```
Season: int
LowTeamID: int
HighTeamID: int
league: str  ("M" or "W")
low_seed: int
high_seed: int
seed_diff: int
low_elo: float
high_elo: float
elo_diff: float
low_win_pct: float
high_win_pct: float
win_pct_diff: float
low_avg_margin: float
high_avg_margin: float
margin_diff: float
low_off_eff: float (nullable)
high_off_eff: float (nullable)
low_def_eff: float (nullable)
high_def_eff: float (nullable)
low_sos: float
high_sos: float
low_massey_rank: float (nullable, M only)
high_massey_rank: float (nullable, M only)
round_prior: float
round_group: str nullable  ("R1" for Round of 64, "R2+" for Round of 32 through Championship; derivation below)
outcome: int (1 if LowTeam won, 0 otherwise; null for submission rows)
```

### 3.3 Engineering Rules

- All features computed on regular-season data only (no tourney data for current season)
- Features derived from `data/interim/` joins, not directly from raw
- Every feature file must have a corresponding data manifest entry
- Features that are null for women must be handled explicitly (not imputed with men's values)

### 3.4 `round_group` Derivation

`round_group` is derived from the bracket topology using the canonical seed-pair lookup (`data/tourney_round_lookup.csv`) and the assigned seeds for the target season. It is fully deterministic once Selection Sunday seeds are published.

**For training rows (historical seasons):** `round_group` is set from the actual round of the played game.

**For submission rows (all pairs):**
- If both teams have a seed in the target season, look up their earliest possible meeting in the bracket slot tree. That slot's round determines `round_group`.
  - Pairs that meet in R64: `"R1"`
  - Pairs that could only meet in R32 or later: `"R2+"`
- If one or both teams do not have a seed (not in the tournament), `round_group` is `null`. These matchups are definitively unplayed — they will never be scored. They still require a prediction in the submission file; use the model's default output for that pair.

**Consequence for round-group models:** No submission row will have an ambiguous `round_group`. All in-tournament pairs have a deterministic earliest-meeting round. Non-tournament pairs get `null` and are routed to the unified fallback model (or any model — their predictions are never scored).

---

## 4. Model Architecture Decisions

### 4.1 Candidate Architectures

| Tier | Architecture | Rationale | Priority |
|---|---|---|---|
| Baseline | Logistic Regression (seed diff only) | Interpretable floor | Must have |
| Baseline+ | Logistic Regression (all Phase A+B features) | Strong simple model | Must have |
| Primary | Gradient Boosted Trees (XGBoost/LightGBM) | Captures interactions | High |
| Primary | Random Forest | Decorrelated diversity for ensemble | Medium |
| Advanced | Neural network (shallow, 2-layer MLP) | Non-linear calibration | Low |
| Specialist | Round-group models (R1 vs R2+ split; see §4.3) | Exploit structural difference between guaranteed and conditional games | Medium |

### 4.2 M/W Modeling Strategy

**Decision (ADR 0004): Option A — Separate models per league, merged at submission time.**

See `docs/decisions/0004-men-women-tournament-modeling-strategy.md` for full rationale. Summary:
- Women's tournament has different parity profiles and narrower feature availability (no Massey ordinals, no coach records).
- Independent tuning and calibration per league; merge via CSV concatenation at submission assembly.
- All experiment IDs carry an explicit `league` tag (`M` or `W`).

### 4.3 `adj_quality_gap_v10` Challenger Track

`docs/adj_quality_gap_v10.md` describes a strong reference implementation that materially overlaps with this roadmap. Treat it as a **benchmark challenger workstream inside PLAN-002**, not as a replacement for this plan's evaluation policy.

**Comparable benchmark rule:** use the documented **LOSO played-game Brier** results as the standard to beat. Do **not** use its Stage 1 all-matchups score as the primary offline benchmark, because that is a different evaluation universe from this plan's held-out played-game flat Brier.

**What overlaps with PLAN-002:**
- separate men/women models
- Elo as a primary strength signal
- SOS-adjusted efficiency as a core quality idea
- blending multiple probability sources
- leave-future-out historical evaluation

**What the challenger adds beyond current repo implementation:**
- KenPom-style iterative SOS adjustment for `AdjO` / `AdjD`
- women-only home-court correction before SOS iteration
- richer situational feature set, especially for men
- tuned Elo with season carryover, MOV weighting, and separate regular/tourney weights
- men-specific margin regression, probability conversion, and dynamic temperature scaling

**Planning stance:** every `adj_quality_gap_v10` component must be added as a named challenger experiment and compared against the frozen leaders (`ARCH-01` men, `ARCH-04` women). Create a new plan only if this grows into a full standalone replication effort with its own gates and timeline.

### 4.4 Round-Group Modeling Strategy

**Open question (to be resolved by experiment):** Does splitting training and/or prediction by round group improve held-out flat Brier, especially on guaranteed `R1` games, over a single unified model?

**Round group definitions:**
- `R1`: Round of 64 only — guaranteed games; largest sample of high-leverage matchups; seed-based priors strongest here
- `R2+`: Round of 32 through Championship — lower sample per round; upset rates shift; strength signals matter more relative to seed priors

**Hypothesis:** R1 games are structurally different (high certainty for top seeds, uniform upset risk for mid-seeds) from R2+ games (cumulative attrition, fatigue, matchup-specificity). A model trained exclusively on R1 may be better calibrated on R1 games than one trained on all rounds together.

**Counter-hypothesis:** Sample size within a single round is small (~32 games per season per league). Splitting by round group reduces training data per model and may increase variance more than it reduces bias.

**Experiment gate:** Do not split by round group unless ARCH-RG experiments (§9.1) show flat-Brier improvement on `R1` without meaningful degradation on `R2+` in 2023–2024 CV. The unified per-league model is the default.

### 4.5 Architecture Experiment Gates

Before advancing a model family to ensemble candidacy:
1. Brier score < seed-diff logistic baseline on 2023–2024 CV
2. Reliability diagram within 0.05 of diagonal in all bins
3. MLflow run exists with hypothesis tag, params, and CV metric
4. No 2025 holdout overfitting (CV vs 2025 delta < 0.010)

---

## 5. Feature Selection

### 5.1 Selection Protocol

**Step 1 — Univariate filter:**
Compute per-feature Brier improvement vs 0.5 constant baseline across CV folds. Discard features with zero or negative improvement.

**Step 2 — Multivariate permutation importance:**
Fit full feature set model; compute permutation importance on CV validation folds. Rank features.

**Step 3 — Recursive feature elimination:**
RFE using leave-seasons-out temporal CV (same split as §2.1) to find the minimal feature set within 0.002 Brier of the full feature set. Do not use k-fold CV here — shuffling observations across seasons would allow future season data to appear in training folds, constituting leakage. The 0.002 threshold is ~1% relative improvement at the expected operating range of ~0.200; confirm this is still a meaningful threshold once baselines are established.

**Step 4 — Collinearity check:**
VIF > 10 triggers removal of the weaker feature (per permutation importance ranking).

**Step 5 — Leakage audit:**
Manual review of selected feature set against leakage checklist before any submission.

### 5.2 Leakage Checklist

- [ ] No feature derived from current-season tournament results
- [ ] Elo uses only games with DayNum < 134
- [ ] No score or outcome from a game that occurs after Selection Sunday
- [ ] Round-specific priors use only historical seasons (not current-season tourney)
- [ ] Massey ordinals filtered to `RankingDayNum ≤ 133` (no post-selection rankings used in features)
- [ ] Stage 2 predictions use Stage 2 data only (not replayed against Stage 1 universe)

---

## 6. Calibration and Ensemble Stack

### 6.1 Calibration Strategy

**Primary calibration method:** Isotonic regression (trained on held-out CV folds)
**Fallback:** Platt scaling (logistic sigmoid fit on raw model outputs)
**Do not calibrate on training data** — always use out-of-fold predictions

**Calibration validation:**
- Reliability diagram for each league (M and W separately)
- Expected Calibration Error (ECE) < 0.02 target
- Calibrated output must be re-clipped to [0.025, 0.975] after calibration

### 6.2 Ensemble Construction

**Ensemble tiers (in order of complexity):**

```
Tier 1 — Weighted average (simple blend)
  - Weights: uniform or grid-searched on CV
  - Members: seed-diff logistic + Elo logistic + GBT

Tier 2 — Stacked ensemble
  - Level-0: seed-diff logistic, Elo logistic, GBT, RF
  - Level-1: logistic regression (no intercept) on OOF predictions
    Rationale for no intercept: if level-0 models are well-calibrated, the level-1
    should only reweight them — an intercept would shift the overall probability
    distribution away from calibrated outputs. Tier 3 isotonic calibration handles
    any remaining shift; double-shifting here would fight the calibration layer.
  - Meta-features: OOF probabilities only (no raw features at level 1)

Tier 3 — Calibrated stack
  - Apply isotonic calibration to Tier 2 output
  - Final clipping applied
```

**Ensemble construction rules:**
- For stacking, `OOF` means predictions generated only from out-of-season validation folds under the leave-seasons-out scheme. Do not create a second random-fold OOF layer inside the training seasons unless the plan is explicitly amended.
- Level-1 trained on out-of-fold predictions only
- No level-1 model sees the test set during fitting
- Ensemble weights must be explainable (not black-box search without CV grounding)
- Log all ensemble configurations as separate MLflow runs

### 6.3 Diversity Audit

Before finalizing ensemble membership, check pairwise correlation of OOF predictions. Target: no two members with r > 0.95. High-correlation members are redundant and should be replaced.

---

## 7. Tournament-Aware Evaluation

### 7.1 Bracket Simulator and Round-Aware Diagnostics

**Build this early, but do not block baselines on it.** The bracket simulator is a diagnostic layer that helps explain where a model is strong or weak in bracket space. The canonical selection metric remains flat Brier on held-out played tournament games. Baseline experiments may proceed before `scripts/bracket_dp.py` exists; bracket diagnostics become required before advancing beyond baseline model comparison.

#### Canonical selection metric

Use flat Brier on held-out played tournament games as the primary model-selection metric:

```
BS = (1/N) * Σ (p_i - y_i)²
```

where each `i` is a historical tournament game that actually occurred in the held-out fold.

This matches the competition scoring rule most directly and avoids introducing model-dependent weighting into offline selection.

#### Round-aware diagnostic metrics

To reflect the fact that `R1` games are guaranteed and later rounds are progressively less likely, report these diagnostics alongside flat Brier:

- `R1` Brier: flat Brier restricted to Round of 64 games
- `R2+` Brier: flat Brier restricted to Round of 32 through Championship games
- Men/Women split Brier: separate Brier by league
- Optional fixed-weight round-aware Brier using exogenous round weights, for example:

```
FWB = Σ_r w_r * mean((p_i - y_i)² for games in round r)
      ─────────────────────────────────────────────────
                       Σ_r w_r
```

where `w_r` is fixed before evaluation and does not depend on the candidate model. Example starting weights:

- `R1 = 1.0`
- `R2 = 0.5`
- `Sweet16 = 0.25`
- `Elite8 = 0.125`
- `Final4 = 0.0625`
- `Championship = 0.03125`

**Note on weighting semantics:** These weights are applied to the *per-round mean* Brier, not per-game. This differs from the flat Brier, which weights per-game (R1 contributes 32 games vs 1 for the Championship, a natural 32:1 ratio). The FWB formula above applies a 32:1 weight ratio at the extremes (1.0 vs 0.03125) to round means — which collapses the 32-game vs 1-game imbalance into a per-round comparison. Flat Brier naturally handles this via game counts; FWB provides a different, more explicit round-level sensitivity analysis.

These weights are diagnostics only. They may inform prioritization, but they do not replace the canonical flat-Brier target.

#### Bracket DP — play probability computation

```
P(team t wins slot S) = computed by dynamic programming over bracket topology

For a slot S with left child L and right child R:
  P(t wins S) = sum over all opponents o:
                  P(t wins L) * P(o wins R) * P(t beats o)

Base case: seeded slots — P(t wins slot) = 1 if t is assigned that seed slot, else 0

P(t1 meets t2 in slot S) = P(t1 wins left subtree leading to S)
                           * P(t2 wins right subtree leading to S)
```

**Implementation:**

```python
# scripts/bracket_dp.py  (build during Gate 0; required before bracket-diagnostic review)
# Inputs:
#   - MNCAATourneySlots.csv  (bracket topology)
#   - MNCAATourneySeeds.csv  (seed -> TeamID for current season)
#   - matchup_probs: dict[(TeamID_low, TeamID_high)] -> float  (model output)
# Outputs:
#   - play_prob: dict[(TeamID_i, TeamID_j)] -> float  (P of meeting)
#   - round_summary: per-round and per-bucket diagnostic breakdown
# Notes:
#   - Play-in (First Four) slots excluded from DP root
#   - Run separately for M and W brackets
#   - Deterministic given matchup_probs; no Monte Carlo needed for play probabilities
```

#### PlayProb bucket definitions

```
definite    (P ≥ 0.99): R64 — 32 games per bracket; always played
very_likely (P ≥ 0.30): R32 — top seeds very likely to advance
likely      (P ≥ 0.10): Sweet 16 range
plausible   (P ≥ 0.03): Elite 8 / Final 4 range
remote      (P < 0.03): Championship and long-shot cross-bracket paths
```

**Why this matters:** A model that is well-calibrated on guaranteed games but weaker on remote hypothetical paths may still be a strong competition entry. The bucket decomposition helps allocate attention without changing the primary evaluation target.

**Bucket Brier decomposition — report for every experiment:**

| Bucket | Games (approx) | Diagnostic importance | Implication |
|---|---|---|---|
| definite | 32 | Highest | Calibration here is the first priority |
| very_likely | ~20 | High | Second priority for calibration effort |
| likely | ~10 | Moderate | Marginal improvement zone |
| plausible | ~5 | Low | Lower leverage |
| remote | many | Very low | Do not over-optimize |

**Simulator parameters:**
- DP is deterministic; no Monte Carlo required for play probabilities
- Log `flat_brier`, `r1_brier`, `r2plus_brier`, and per-bucket diagnostics to MLflow for every experiment run
- Run separately for M and W brackets; report combined flat Brier as the final summary metric, with bracket diagnostics as supporting analysis

### 7.2 Competitive Differentiation Strategy

**Core tension:** Everyone has access to the same seeds and Massey ordinals. Differentiation comes from:

1. **Better calibration, not just ranking.** A model that says 0.72 when the true probability is 0.72 outscores one that says 0.85 on the same matchups — even with identical rankings.

2. **Women's tournament edge.** Women's tournament has less public modeling attention. More effort on women's calibration may disproportionately improve overall score.

3. **Upset probability modeling.** Brier score rewards accurate low-seed upset probabilities. Do not cap all upsets at a naive floor; quantify historically justified upset rates.

4. **Massey ordinal diversity (men).** With ~100 ranking systems available, systematic extraction of consensus vs. dissent signal may capture information beyond seeds.

5. **Avoid over-engineering.** The majority of Brier improvement comes from the first 80% of feature engineering. Diminishing returns set in fast. Ship a working calibrated model before pursuing exotic features.

**Differentiation priority order:**
1. Calibrate seed-diff baseline (guaranteed above-floor)
2. Add Elo + efficiency (most reliable secondary signals)
3. Calibrate M and W separately
4. Add Massey consensus for men
5. Ensemble two uncorrelated models
6. Tournament-aware late-stage tuning

---

## 8. Submission Strategy

### 8.1 Timeline and Approach

**Single Stage 2 submission after Selection Sunday.** All model development, evaluation, and selection is done offline using historical seasons as the validation set. No incremental Kaggle submissions are used for model feedback — the leaderboard is not part of the development loop.

| Milestone | Date | Action |
|---|---|---|
| Selection Sunday | ~2026-03-15 | Bracket seeds published; Stage 2 sample submission released |
| Stage 2 feature build | 2026-03-15–16 | Compute all 2026 regular-season and matchup features required by the selected frozen model family. At minimum this includes Phase A/B inputs; if the chosen model depends on Phase C or D features, build those exact inference-time features as well. |
| Stage 2 prep | 2026-03-16–18 | Join 2026 features with 2026 seeds; derive `round_group` and any other bracket-derived inference features required by the selected model; apply the frozen pre-fitted model to Stage 2 matchup pairs |
| Final submission | 2026-03-18–19 | Upload best model output; select it explicitly before deadline |
| Hard deadline | 2026-03-19 16:00 UTC | Submission must be selected in Kaggle UI |

**Pre-Selection Sunday work (2026-03-10 to 2026-03-15):** Complete Gates 0–2 entirely. All feature engineering, model training, calibration, and ensemble construction uses historical data. The Stage 2 run is a final application of the already-selected model — not a new training run. The only allowed Stage 2 changes are inference-time feature generation for 2026 data and bracket-derived routing fields required by the chosen frozen model family.

### 8.2 Submission Types

| Type | When | Description |
|---|---|---|
| Validation run | Pre-Selection Sunday | Offline scoring on 2023–2025 held-out seasons to select model |
| Stage 2 submission | Post-Selection Sunday | Apply selected model to Stage 2 pairs; upload once |
| Final selection | 2026-03-19 | Confirm submission is selected in Kaggle UI |

### 8.2.1 Gate 3 Freeze

**Frozen model choices entering Gate 3 (2026-03-12):**

- Men: `ARCH-01` seed-diff logistic baseline
  - MLflow run: `c0ef5d21cbf44cc185edc4ec7b920b43`
  - 2023-2024 flat Brier: `0.197508`
  - 2025 holdout run: `c3733a50efc7491586f3060e0dc317b0`

- Women: `ARCH-04B` tuned carryover Elo + seed logistic baseline
  - MLflow run: `ba0a02b8c8bc404089bdd2de7fe9a917`
  - 2023-2024 flat Brier: `0.132193`
  - 2025 holdout run: `f6eac553495949a5805c84fb4cdef757`

**Gate 3 operating rule:** these are the submission-default models unless a later Gate 3 challenger beats them on the same held-out flat-Brier protocol. No model is promoted during Stage 2 inference preparation for subjective reasons, additional complexity, or leaderboard curiosity alone.

### 8.3 Submission Assembly Checklist

Before each submission:
- [ ] All TeamIDs in submission match sample submission exactly
- [ ] No missing rows
- [ ] All Pred values are floats in [0.025, 0.975]
- [ ] ID format is correct (`YYYY_LowID_HighID`)
- [ ] Submission was generated from a committed and logged MLflow run
- [ ] Submission file is saved to `data/submissions/` with descriptive name and date
- [ ] Run `uv run python scripts/validate_submission.py <file>` before upload

### 8.4 Emergency Protocol

If the Stage 2 pipeline breaks after Selection Sunday:
1. Fall back to the seed-difference logistic baseline — it is always recoverable in <1 hour given that seeds are published
2. Do not attempt untested code changes under deadline pressure
3. The seed-diff model is the guaranteed floor; any pipeline complexity beyond that is optional

---

## 9. Experiment Design Master Matrix

### 9.1 Architecture Experiments

| ID | Name | Model Family | Features | League | Hypothesis | Priority |
|---|---|---|---|---|---|---|
| ARCH-01 | Seed-diff logistic (M) | Logistic Regression | seed_diff | M | Seed difference alone is the strongest single predictor | P0 |
| ARCH-02 | Seed-diff logistic (W) | Logistic Regression | seed_diff | W | Same hypothesis, women's data | P0 |
| ARCH-03 | Elo logistic (M) | Logistic Regression | elo_diff, seed_diff | M | Elo adds signal beyond seeds | P0 |
| ARCH-04 | Elo logistic (W) | Logistic Regression | elo_diff, seed_diff | W | Same, women's data | P0 |
| ARCH-05 | Multi-feature logistic (M) | Logistic Regression | Phase A+B features | M | Richer feature set improves Brier vs seeds alone | P1 |
| ARCH-06 | Multi-feature logistic (W) | Logistic Regression | Phase A+B features | W | Same, women's data | P1 |
| ARCH-07 | GBT (M) | XGBoost/LightGBM | Phase A+B+C features | M | Tree model captures nonlinear feature interactions | P1 |
| ARCH-08 | GBT (W) | XGBoost/LightGBM | Phase A+B+C features | W | Same, women's data | P1 |
| ARCH-09 | Random Forest (M) | Random Forest | Phase A+B features | M | RF provides uncorrelated diversity for ensemble | P2 |
| ARCH-10 | Round-specific logistic (M) | Logistic Regression | seed_diff + round_prior | M | First-round vs late-round priors differ significantly | P2 |
| ARCH-10-W | Round-specific logistic (W) | Logistic Regression | seed_diff + round_prior | W | Same hypothesis, women's data; women's round priors may differ from men's due to different parity structure | P2 |
| ARCH-11 | Calibrated GBT (M) | XGBoost + isotonic | Phase A+B+C | M | Calibration wrapper improves Brier over raw GBT | P1 |
| ARCH-12 | Calibrated GBT (W) | XGBoost + isotonic | Phase A+B+C | W | Same, women's data | P1 |
| ARCH-RG-01 | R1-only logistic (M) | Logistic Regression | Phase A+B features | M | R1-trained model is better calibrated on guaranteed Round of 64 games than a full-round model | P2 |
| ARCH-RG-02 | R1-only logistic (W) | Logistic Regression | Phase A+B features | W | Same, women's data | P2 |
| ARCH-RG-03 | R2+-only logistic (M) | Logistic Regression | Phase A+B+C features | M | R2+-trained model captures different signal distribution (strength > seed) | P2 |
| ARCH-RG-04 | R2+-only logistic (W) | Logistic Regression | Phase A+B+C features | W | Same, women's data | P2 |
| ARCH-RG-05 | Round-group blend (M) | R1 model + R2+ model, blended by round_group | Phase A+B+C | M | Blending round-group models improves held-out flat Brier by specializing on guaranteed vs later-round games | P2 |
| ARCH-RG-06 | Round-group blend (W) | R1 model + R2+ model, blended by round_group | Phase A+B+C | W | Same, women's data | P2 |

### 9.2 Feature Experiments

| ID | Name | Feature Added | Baseline Model | Hypothesis | Priority |
|---|---|---|---|---|---|
| FEAT-01 | Elo vs seed diff | elo_diff | ARCH-01 | Elo explains variance seed difference misses | P0 |
| FEAT-02 | Win percentage | win_pct_diff | ARCH-03 | W% adds signal beyond Elo for current-season form | P1 |
| FEAT-03 | Adjusted efficiency | off_eff, def_eff matchup | ARCH-05 | Efficiency captures pace-adjusted quality | P1 |
| FEAT-04 | Massey consensus rank | massey_rank_diff | ARCH-05 | Consensus of 100 ranking systems > any single system | P1 |
| FEAT-05 | Strength of schedule | sos_diff | ARCH-05 | SOS contextualizes W% for teams in weak conferences | P2 |
| FEAT-06 | Recent form (last 10 games) | recent_win_pct_diff | ARCH-05 | Hot teams outperform season-long metrics suggest | P2 |
| FEAT-07 | Coach experience | coach_tourney_games | ARCH-07 | Experienced coaches outperform in close tourney games | P2 |
| FEAT-08 | Tempo interaction | tempo_diff, tempo_product | ARCH-07 | Fast-vs-slow matchups have different upset rates | P3 |
| FEAT-09 | Historical seed-pair rate | seed_pair_historical_rate | ARCH-01 | Granular seed pair (not just diff) carries extra signal | P2 |
| FEAT-10 | Conference strength | conf_elo_diff | ARCH-05 | Conference quality adjusts for schedule strength | P3 |
| FEAT-11 | SOS-adjusted net efficiency | sos_adjusted_net_eff_diff | ARCH-03 / ARCH-04 challenger | Combined efficiency + schedule strength improves Brier over raw Elo baselines | P1 |
| FEAT-12 | Iterative KenPom-style adjusted efficiency | adj_qg_diff | `adj_quality_gap_v10` challenger | Iterative opponent-adjusted efficiency beats the simpler SOS-adjusted net-efficiency proxy | P1 |
| FEAT-13 | Women HCA-corrected adjusted efficiency | women_hca_adj_qg_diff | `adj_quality_gap_v10` challenger | Pre-SOS home-court correction improves women's adjusted quality signal | P1 |
| FEAT-14 | Men situational box-score differentials | d_AstRate, d_FTR, d_TOVr, d_ConfTourneyWR, d_CloseWR1 | `adj_quality_gap_v10` challenger | Richer situational signals improve men's quality model | P2 |
| FEAT-15 | Women close-game performance | d_CloseWR3 | `adj_quality_gap_v10` challenger | Women's 3-point close-game win rate adds signal beyond Elo and quality gap | P2 |

### 9.3 Combination and Validation Experiments

| ID | Name | Components | Validation Method | Hypothesis | Priority |
|---|---|---|---|---|---|
| COMBO-01 | Simple blend (M) | ARCH-03 + ARCH-07 | Leave-seasons-out CV | Blending Elo logistic and GBT improves vs either alone | P1 |
| COMBO-02 | Simple blend (W) | ARCH-04 + ARCH-08 | Leave-seasons-out CV | Same, women's data | P1 |
| COMBO-03 | Stacked ensemble (M) | ARCH-03 + ARCH-07 + ARCH-09 | OOF stacking from out-of-season-fold predictions only | Level-1 meta-learner improves on best single model | P2 |
| COMBO-04 | Stacked ensemble (W) | ARCH-04 + ARCH-08 | OOF stacking from out-of-season-fold predictions only | Same, women's data; RF omitted because narrower W feature set makes a third member redundant — add ARCH-09-W if W flat Brier is consistently higher than M | P2 |
| COMBO-05 | Full calibrated ensemble | COMBO-03 + COMBO-04, per-league isotonic (M and W calibrated separately before concatenation) | Reliability diagram + 2025 holdout | End-to-end calibrated system beats uncalibrated | P1 |
| COMBO-06 | M+W combined submission | COMBO-05 (M) + COMBO-05 (W) | Submission schema validation | Combined submission is valid and best available | P0 |
| COMBO-RG-01 | Round-group blend vs unified (M) | ARCH-RG-05 vs best unified M model | Flat Brier with R1/R2+ breakdown, 2023–2024 CV | Round-group blend improves `R1` Brier without degrading overall flat Brier | P2 |
| COMBO-RG-02 | Round-group blend vs unified (W) | ARCH-RG-06 vs best unified W model | Flat Brier with R1/R2+ breakdown, 2023–2024 CV | Same, women's data | P2 |
| COMBO-RG-03 | Round-group blend in full stack (M) | ARCH-RG-05 replacing ARCH-07 in COMBO-03 | Flat Brier overall, 2025 holdout | Substituting round-group model into ensemble improves overall flat Brier | P3 |
| VAL-01 | 2025 holdout sanity check | Best current model | 2025 tournament outcomes | CV score generalizes to unseen season | P0 (every model) |
| VAL-02 | Calibration audit | Best current model | Reliability diagram | Output probabilities match empirical win rates | P0 (every model) |
| VAL-03 | Bracket simulator diagnostics | Best current model | Deterministic bracket DP | Bracket structure does not expose unexpected calibration risk in high-likelihood buckets | P1 |
| VAL-04 | Women's parity check | ARCH-06 vs ARCH-08 | CV by seed pair | Women's model appropriately reflects higher parity | P1 |
| VAL-05 | `adj_quality_gap_v10` benchmark comparison | Best local challenger vs documented benchmark | LOSO played-game Brier | Local challenger approaches or exceeds the documented benchmark on the comparable metric | P1 |

---

## 10. Execution Checklist and Gates

### Gate 0 — Infrastructure Ready (by 2026-03-11)
- [ ] `scripts/bracket_dp.py` implemented and tested (required before bracket-diagnostic review; not a blocker for baseline flat-Brier experiments)
- [ ] Flat Brier, `R1` Brier, and `R2+` Brier computed for at least one historical season as smoke test
- [ ] Feature pipeline produces Phase A feature table
- [ ] CV framework runs with leave-seasons-out splits, reporting flat Brier as primary metric and round-aware diagnostics as secondary outputs
- [ ] `scripts/validate_cv.py` implemented (see §2.5 for spec: per-season Brier, calibration curve, reliability diagram, MLflow artifact logging)
- [ ] Submission writer passes schema validation
- [ ] `scripts/validate_submission.py` implemented (see §8.3: schema, missing rows, Pred range, ID format)
- [ ] MLflow tracking wired for all runs (logging `flat_brier`, `r1_brier`, `r2plus_brier`, and per-bucket diagnostics)
- [ ] Baseline ARCH-01 and ARCH-02 logged with flat Brier and round-aware diagnostics

### Gate 1 — Baselines Established (by 2026-03-13)
- [ ] ARCH-01 through ARCH-04 complete with CV flat Brier and round-aware diagnostics
- [ ] 2025 holdout sanity check (VAL-01) complete
- [ ] Feature Phase B complete

### Gate 2 — Primary Models Ready (by 2026-03-16)
- [ ] ARCH-05 through ARCH-08 complete
- [ ] FEAT-01 through FEAT-04 evaluated
- [ ] COMBO-01 and COMBO-02 complete
- [ ] Calibration audit (VAL-02) passed for each model candidate

### Gate 3 — Ensemble and Final Submission (by 2026-03-18)
- [ ] Gate 3 freeze recorded and unchanged unless a later challenger beats the frozen leader on held-out flat Brier
- [ ] Men frozen leader: `ARCH-01`
- [ ] Women frozen leader: `ARCH-04B`
- [ ] COMBO-05 and COMBO-06 complete
- [ ] VAL-03 bracket DP diagnostics run on final ensemble
- [ ] Per-bucket and `R1/R2+` Brier decomposition reviewed; no unexpected degradation on guaranteed `R1` games
- [ ] Final submission candidate selected and validated
- [ ] Submission checklist passed
- [ ] 2026 inference feature tables validated for the selected model family: no missing rows for tournament teams, all required seeds/features mapped, and feature ranges within historical bounds
- [ ] `mlruns/` backed up to separate location

### Hard Deadline — 2026-03-19 16:00 UTC
- [ ] Final submission selected in Kaggle UI
- [ ] Submission file archived in `data/submissions/` with final tag

---

## 11. Related Documents

- `docs/COMPETITION_OPERATIONS.md` — submission mechanics and timeline
- `docs/data/FILE_CATALOG.md` — full data inventory
- `docs/data/RELATIONSHIP_DIAGRAM.md` — join-key reference
- `docs/data/TOURNEY_ROUND_ASSIGNMENT.md` — round assignment policy
- `docs/decisions/0004-men-women-tournament-modeling-strategy.md` — accepted M/W split decision
- `docs/experiments/experiment-log.md` — ongoing experiment notes
- `docs/roadmaps/MODELING_ROADMAP.md` — high-level stage roadmap
- `docs/roadmaps/roadmap.md` — canonical working roadmap (update as tasks complete)
- `MASTER_CHECKLIST.md` — operational session checklist
