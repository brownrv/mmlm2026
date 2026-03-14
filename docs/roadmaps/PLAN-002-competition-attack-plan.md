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

**Underdog threshold — when aggressive picks have positive expected value.**

Brier score is strictly proper, so the optimal prediction is always the true probability. However, when evaluating whether a calibrated model output for a moderate underdog is worth preserving (rather than pulling toward 0.5), the following closed-form result applies:

The expected Brier score gain from predicting an underdog at their true probability p, relative to the naive 0.5 baseline, is maximised at **p = 1/3 ≈ 33.3%**:

```
Maximize  f(p) = p·(1 − p)²   over p ∈ [0, 1]
f′(p) = (1 − p)² − 2p(1 − p) = (1 − p)(1 − 3p) = 0
→  p* = 1/3
```

**Practical interpretation:**
- If the model outputs a calibrated probability of 0.35–0.48 for an underdog, the math confirms that keeping that output (or making it slightly more aggressive) has positive expected Brier value relative to pulling it toward 0.5.
- If the model outputs < 0.33 for an underdog, aggressive deviation toward that team is expected to *lose* in expectation; trust the calibrated output.
- This is not a license to pick random upsets — it defines the probability zone (> 1/3 true win chance) where the model's calibrated output should be respected rather than smoothed away.
- The symmetric corollary: for strong favorites (true probability > 2/3), resisting the urge to pull predictions back toward 0.5 is equally mathematically grounded.
- This is not a license to deviate from the model's calibrated output — it is a principled defense of that output against post-hoc smoothing toward 0.5, which itself is a deviation from strict propriety.

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
| Conference membership | `MTeamConferences.csv` | `WTeamConferences.csv` | Season-by-season team–conference mapping; required for COOPER-ARCH-03 (conference mean Elo reversion) |

**Data constraints:**

- `data/raw/` is immutable — never modify
- Canonical round assignment: `data/tourney_round_lookup.csv` (seed-pair, not DayNum)
- Canonical team name mapping: `data/TeamSpellings.csv`
- Elo regular-season cutoff: `DayNum < 134` (rule still valid; separate from round assignment)

### 1.3.1 Supplemental Local Datasets for Late Challengers

The repository now contains several non-Kaggle datasets that are explicitly in scope for late challenger work. These are not default production dependencies yet; they are challenger assets and must beat the frozen leaders on held-out flat Brier before they are promoted.

| Dataset | Path | Coverage | Best use | Main risks |
|---|---|---|---|---|
| Reference all-matchups benchmark table | `data/reference/` | Men 2003–2025, Women 2010–2025 | Compare local models against `adj_quality_gap_v10`, inspect bucket/round-specific misses, and build challenger-targeted diagnostics | Contains model-derived features and probabilities from the benchmark model; use for comparison, error analysis, and feature inspiration, not as a leakage-prone direct training target |
| BetExplorer odds | `data/processed/betexplorer/` | Men and women, regular season and tournament, opening/closing odds | Market-implied priors, calibration anchors, and late-round matchup strength features | Coverage gaps, team-name mapping quality, and market data availability may differ by league and season |
| ESPN parsed game and player data | `data/processed/espn/` | Men and women historical seasons in parquet | Richer box-score and player-level strength signals, lineup-agnostic form, possession decomposition, and situational features | Join complexity, schema drift, and the risk of over-building low-signal features under deadline |
| External tournament-progression forecasts | `data/processed/tourney_forecasts/` (populate post-Selection Sunday if secured) | Men and women, target season only; **preferred source: COOPER (Silver Bulletin)** — injury-adjusted, bracket-simulates 'hot' (ratings update during simulation), internal ratings blended 5/8 COOPER + 3/8 KenPom (M) / Her Hoop Stats (W); fallback sources: ESPN bracket forecasts, T-Rank, teamrankings.com; columns `rd1_win`–`rd6_win` per team; 538 no longer publishes as of 2024 | `LATE-EXT-04` challenger: blend as direct H2H prior (via `goto_conversion`), derive conditional-strength features (`rd_r_win / rd_{r-1}_win`), or use as calibration anchor | Data availability not guaranteed for 2026; use only forecasts from the earliest available pre-tournament date (before Selection Sunday; no updating with realized results); team-name mapping to Kaggle TeamID required; women's COOPER launched March 10 — verify full M+W team coverage before Selection Sunday |

**Planning stance for these datasets:**
- `data/reference/` is primarily an analysis and benchmark-comparison asset. It is best used to identify where local models differ from the benchmark on the same matchup universe and to prioritize challenger hypotheses.
- `data/processed/betexplorer/` is the highest-upside late challenger dataset if coverage is clean, because market odds can act as a strong compressed prior for strength and calibration.
- `data/processed/espn/` is the best source for improving internal strength ratings and situational features, especially if the objective is a better latent team-quality model rather than a broader final classifier.
- `data/processed/tourney_forecasts/` is **conditional on data availability confirmed post-Selection Sunday (March 15)**. If a clean source with full M+W team coverage is secured, this is the highest-leverage remaining external signal because it is bracket-aware and orthogonal to all internal Elo/GBT features. **Preferred source: COOPER (Silver Bulletin)** — injury-adjusted bracket simulations that run 'hot' (team ratings update during simulation based on simulated game results, giving a more realistic posterior for later rounds); blends COOPER 5/8 + KenPom/Her Hoop Stats 3/8. If COOPER is not accessible, fall back to ESPN, T-Rank, or teamrankings.com. If data cannot be secured by 2026-03-16, skip LATE-EXT-04 entirely and proceed with the frozen leaders.

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
| Market-implied probability / spread prior | + stronger market favorite wins | Medium | Late-challenger only; requires strict pregame timing, clean team mapping, and market-coverage audit |
| ESPN-derived four-factor / possession composites | + stronger efficiency profile wins | Medium | Late-challenger only; must be season-stable and aggregated without future leakage |
| Player/rotation continuity proxies | Interaction | Medium | Late-challenger only; useful only if historical coverage and joins are stable |
| GLM team quality coefficients (Bradley-Terry OLS) | + higher quality coefficient wins | None | `glm_quality_diff = α_LowTeamID - α_HighTeamID`; OLS fit on regular-season point diffs, independently per season; see MH7-FEAT-01 in §9.2 for full design matrix spec; requires detailed results |
| Conditional round-progression strength (external forecast) | + higher conditional strength wins | Medium | `cond_r{n}_strength = rd{n}_win / rd{n-1}_win` per team; diff = LowTeam − HighTeam; derived from external tournament-progression forecasts; LATE-EXT-04 only; see GC-FEAT-01 in §9.2; use only forecasts from the earliest available pre-tournament date (before Selection Sunday); requires data availability confirmed post-Selection Sunday |
| Pace (expected combined points per game) | Interaction; modulates confidence in margin-based predictions | None | `low_pace` = OT-normalized rolling average of total game score per game involving LowTeamID (LowTeamID score + opponent score); `high_pace` equivalent; `pace_diff = low_pace − high_pace`; higher-scoring games have higher score variance so a marginal point carries less signal; GBT may learn non-linear interactions between pace and margin/efficiency features; see COOPER-FEAT-01 in §9.2 |
| Opponent raw box-score averages | − higher opponent-score-allowed loses; varies by stat | None | `low_avg_opp_{stat}` / `high_avg_opp_{stat}` / `avg_opp_{stat}_diff` for stat ∈ {Score, FGA, Blk, PF, TO, Stl}; each is the mean stat recorded by opponents *against* that team per regular-season game (OT-normalized); simpler defensive quality proxy complementary to adjusted efficiency; see MH7-FEAT-02 in §9.2 |

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
| External bracket-forecast inputs (tournament-progression probabilities) | Using a bracket forecast published or updated after tournament start (Selection Sunday) | Use only forecasts from the earliest available pre-tournament date (before Selection Sunday); do not update forecast probabilities with realized bracket outcomes |

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
  B6. GLM team quality coefficients (OLS on regular-season point diffs; team-pair indicator design matrix; fit per season; see MH7-FEAT-01)
  B7. Pace factor (OT-normalized rolling avg of total points per game per team; see COOPER-FEAT-01)

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
low_glm_quality: float (nullable; optional late-challenger column; requires detailed results; see MH7-FEAT-01)
high_glm_quality: float (nullable; optional late-challenger column; requires detailed results; see MH7-FEAT-01)
glm_quality_diff: float (nullable; optional late-challenger column)
# MH7-FEAT-02 adds 18 optional late-challenger columns following the same Low/High/diff pattern:
# low_avg_opp_{stat}, high_avg_opp_{stat}, avg_opp_{stat}_diff for stat in {Score, FGA, Blk, PF, TO, Stl}
# These columns are null below each league's training floor (men pre-2010, women pre-2010).
# COOPER-FEAT-01 adds 3 optional late-challenger columns:
# low_pace, high_pace, pace_diff
# low_pace = OT-normalized avg of (team score + opponent score) per regular-season game for LowTeamID
# Available wherever compact results exist; no detailed-results dependency.
# Optional late-challenger columns are omitted from the base feature table until the
# corresponding challenger is promoted via §4.5 gate criteria.
round_prior: float
round_group: str nullable  ("R1" for Round of 64, "R2+" for Round of 32 through Championship; derivation below)
outcome: int (1 if LowTeam won, 0 otherwise; null for submission rows)
```

### 3.3 Engineering Rules

- All features computed on regular-season data only (no tourney data for current season)
- Features derived from `data/interim/` joins, not directly from raw
- Every feature file must have a corresponding data manifest entry
- Features that are null for women must be handled explicitly (not imputed with men's values)
- **Overtime normalization:** When aggregating game-level box-score counting stats (Score, FGA, OR, FTM, FTA, DR, Blk, PF, TO, Stl) across games, divide each game's totals by `adjot = max(1, 1 + NumOT / 5)` before averaging. This prevents teams with many overtime games from appearing artificially stronger in raw counting-stat aggregates. Do not apply to efficiency-derived features already normalized by possessions (off_eff, def_eff).

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

**Planning stance:** every `adj_quality_gap_v10` component must be added as a named challenger experiment and compared against the current frozen leaders (generalization-tuned reference-style margin model for men, `COOPER-ARCH-01 + COOPER-ARCH-04 v1` for women). Create a new plan only if this grows into a full standalone replication effort with its own gates and timeline.

### 4.4 Round-Group Modeling Strategy

**Open question (to be resolved by experiment):** Does splitting training and/or prediction by round group improve held-out flat Brier, especially on guaranteed `R1` games, over a single unified model?

**Round group definitions:**
- `R1`: Round of 64 only — guaranteed games; largest sample of high-leverage matchups; seed-based priors strongest here
- `R2+`: Round of 32 through Championship — lower sample per round; upset rates shift; strength signals matter more relative to seed priors

**Hypothesis:** R1 games are structurally different (high certainty for top seeds, uniform upset risk for mid-seeds) from R2+ games (cumulative attrition, fatigue, matchup-specificity). A model trained exclusively on R1 may be better calibrated on R1 games than one trained on all rounds together.

**Counter-hypothesis:** Sample size within a single round is small (~32 games per season per league). Splitting by round group reduces training data per model and may increase variance more than it reduces bias.

**Experiment gate:** Do not split by round group unless `ARCH-RG` or `LATE-ARCH-RG` experiments (§9.1) show flat-Brier improvement on `R1` without meaningful degradation on `R2+` in 2023–2024 CV. The unified per-league model is the default.

### 4.5 Late Challenger Queue

The original Gates 0–3 establish a disciplined frozen-pair baseline. The queue below governs **late challenger work only**. These challengers are still part of `PLAN-002`; they do not justify a new `PLAN-003` unless the work expands into a separate post-deadline research program with a different success criterion.

**Why this stays in PLAN-002:**
- same competition target
- same held-out flat-Brier evaluation policy
- same frozen leaders as the benchmark to beat
- same operational endpoint: a single Stage 2 submission

**Operating rule for late challengers:** every challenger must beat the current frozen leader for its league on the same held-out flat-Brier protocol before it advances. Use 2025 sanity checks before promotion if the challenger beats the 2023–2024 window.

**Guardrail for `data/reference`:** benchmark probabilities, benchmark-derived matchup likelihood fields, and benchmark feature columns are diagnostic-only assets. They may be used for comparison, gap analysis, bucket targeting, and challenger hypothesis formation, but not as direct supervised targets, leakage-prone training features, or replacement model-selection metrics.

**Current frozen leaders to beat:**
- Men: generalization-tuned reference-style margin model
  - MLflow run: `337d3b992b884dbb800c078561d37622`
  - 2023–2024 flat Brier: `0.195566`
- Women: `COOPER-ARCH-01 + COOPER-ARCH-04 v1` routed women + ESPN four-factor + conference-rank model with COOPER Elo replacement
  - MLflow run name: `cooper-arch-01-04-women`
  - 2023–2024 flat Brier: `0.129201`
  - 2025 sanity-check run name: `val-01-2025-holdout-women-cooper-arch-01-04`

#### Tier 1 — Highest-value late challengers

| Rank | ID | Challenger | League | Expected Payoff | Effort | Why it is next |
|---|---|---|---|---|---|---|
| 1 | LATE-ARCH-RG-07 | Routed `R1` vs `R2+` men model | M | High | Medium | Men still show the clearest persistent `R2+` weakness; calibration-only adjustment was not enough |
| 2 | LATE-ARCH-RG-08 | Routed `R1` vs `R2+` women model | W | Medium | Low-Medium | Women may still benefit from stage-specific coefficient weighting even though calibration-only changes did not win |
| 3 | LATE-RATE-01 | Improved men latent strength model | M | High | Medium-High | The best men gains so far came from better strength estimation, not broader final-model complexity |
| 4 | LATE-RATE-02 | Improved women latent strength model | W | Medium-High | Medium-High | Women may be closer to their ceiling on simple calibration tweaks than on upstream strength quality |

#### Tier 2 — External-data challengers

| Rank | ID | Challenger | League | Expected Payoff | Effort | Primary dataset | Why it matters |
|---|---|---|---|---|---|---|---|
| 5 | LATE-MKT-01 | Market-implied odds prior / calibration feature | M+W | Very High | High | `data/processed/betexplorer/` | Market odds are a compact summary of public strength and matchup information; especially useful for later rounds |
| 6 | LATE-EXT-01 | Men external benchmark-guided challenger | M | Medium-High | Medium | `data/reference/` | Use benchmark all-pairs outputs and features to identify where the local men model still lags and target those cells |
| 7 | LATE-EXT-02 | Women external benchmark-guided challenger | W | Medium | Medium | `data/reference/` | Use benchmark comparison to isolate round/bucket cells where women still trail the reference |
| 8 | LATE-EXT-03 | ESPN-derived advanced situational features | M+W | Medium-High | High | `data/processed/espn/` | Richer possessions, four factors, player/game context, and situational form can improve latent rating quality |
| 9 | LATE-EXT-04 | External tournament-progression forecast challenger | M+W | Very High (if data secured) | Medium | `data/processed/tourney_forecasts/` | Bracket-aware pre-tournament round-progression probabilities (ESPN, T-Rank, KenPom, or equivalent) are orthogonal to all internal Elo/GBT signals; three sub-approaches: (A) direct H2H blend via `goto_conversion`, (B) conditional-strength features (`rd_r_win / rd_{r-1}_win`), (C) calibration anchor; **conditional on data availability confirmed post-Selection Sunday — skip entirely if not secured by 2026-03-16** |

#### Tier 3 — Representation-learning challengers

| Rank | ID | Challenger | League | Expected Payoff | Effort | Why lower priority |
|---|---|---|---|---|---|---|
| 9 | LATE-EMB-01 | Men team embeddings from regular-season game graph | M | Medium | High | Useful if treated as a latent-strength feature generator, but slower to build and validate |
| 10 | LATE-EMB-02 | Women team embeddings from regular-season game graph | W | Medium | High | Same concept, but women likely benefit more first from better upstream data/rating construction |
| 11 | LATE-NN-01 | End-to-end shallow neural tournament model | M+W | Low-Medium | High | Least attractive before deadline; calibration and overfitting risk are high for the likely marginal upside |

#### Tier 4 — Notebook- and methodology-derived feature and training scheme challengers

All challengers in this tier use Kaggle competition data only (no external datasets required). Sources: `notebooks/0-1471-stage-2-metastack-madness-engine-eda.ipynb` and `docs/cooper_ratings.docx` (COOPER methodology, Silver Bulletin). Leakage policy is identical to all other challengers: features computed using only data available before DayNum 134 of the target season, from seasons strictly before the target year.

| Rank | ID | Challenger | League | Expected Payoff | Effort | Why it is next |
|---|---|---|---|---|---|---|
| 12 | LATE-FEAT-21 | Tournament-only Elo | M+W | Medium-High | Low | Second Elo pass on historical tourney games only (separate K-factor, lower season reversion); existing Elo infrastructure reusable; distinct persistent-pedigree hypothesis from regular-season Elo |
| 13 | LATE-FEAT-22 | Elo momentum (end-of-season − DayNum-115 mid) | M+W | Medium | Low | Single derived column from existing Elo at a mid-season breakpoint; tests whether rating trajectory predicts tournament outcomes beyond end-of-season level |
| 14 | LATE-FEAT-26 | Pythagorean expectancy | M+W | Medium | Low | Classic score-based win estimator `pts^10.25 / (pts^10.25 + allowed^10.25)`; diverges from W% for teams winning or losing many close games; trivial to compute from existing results |
| 15 | LATE-FEAT-23 | Seed-Elo gap (over/under-seeded) | M+W | Medium | Low | Cross-signal: `actual_elo − (1750 − (seed−1) × 25)`; teams whose Elo substantially exceeds what their seed implies may be systematically mispriced by seed-based models |
| 16 | LATE-ARCH-DW-01 | Decay-weighted training | M+W | Medium | Low-Medium | Exponential recency weighting (decay per season backwards); training scheme modifier that applies to any existing model without feature changes; §2.1 notes this as a possibility but no experiment exists |
| 17 | LATE-FEAT-25 | Season momentum (H2 − H1 net rating) | M+W | Medium | Low-Medium | Net rating in second half of season minus first half; captures improving vs. declining teams across the full season arc; complements recent-form last-10 which only covers the tail |
| 18 | LATE-ARCH-META-01 | Logit-Ridge meta-learner | M+W | Medium | Low | Replace no-intercept logistic in COMBO-03/04 Tier 2 with Ridge regression in logit space; logit transform emphasizes tail calibration where Brier loss is highest; alpha tuned via leave-seasons-out CV |
| 19 | LATE-FEAT-30 | Massey PCA and disagreement | M | Medium | Medium | PCA on all Massey systems to extract orthogonal consensus components (PC1–PC5); `massey_std` captures ranking disagreement as a calibration uncertainty proxy; men only (Massey not available for women) |
| 20 | LATE-FEAT-28 | Win quality bins (close ≤5 pt, blowout ≥15 pt) | M+W | Low-Medium | Low | Extends CloseWR work (FEAT-14/15 at ≤1/≤3 thresholds); ≤5 captures broader clutch performance; blowout-win rate ≥15 is entirely new and tests whether dominant teams are undervalued by margin-only features |
| 21 | LATE-FEAT-24 | Late-5 form with offensive/defensive split | M+W | Medium | Medium | Separate offensive and defensive net rating over the final 5 pre-tournament games (DayNum ≥ 115); hypothesis: defense travels better than offense to neutral sites; distinct from last-10 window (B3) |
| 22 | LATE-FEAT-27 | Road and neutral-site performance profiles | M+W | Medium | Medium | Disaggregate regular-season games by WLoc ('H'/'A'/'N'); compute win%, margin, and game count for road and neutral contexts separately; neutral-site win% is the most tournament-relevant form signal |
| 23 | LATE-FEAT-29 | Conference percentile rank | M+W | Low-Medium | Medium | Team's percentile rank within its own conference by net rating; distinguishes a dominant mid-major from a fringe power-conference team better than raw conference average strength (FEAT-10) |
| 24 | LATE-FEAT-31 | Tournament program pedigree | M+W | Low-Medium | Low-Medium | 5-year lookback: NCAA tournament appearances count, average seed, best seed; team-level organizational experience beyond coach features (FEAT-07); requires only historical `{M\|W}NCAATourneySeeds.csv` |
| 25 | LATE-ARCH-CB-01 | CatBoost base learner | M+W | Low-Medium | Medium | Additional base learner diversity beyond LightGBM/XGBoost; ordered boosting and built-in categorical handling; reference notebook shows CatBoost as highest meta-weight member (0.359 for men); only warranted if diversity audit shows high OOF correlation among current ensemble members |
| 26 | COOPER-ARCH-01 | Win-bonus Elo (winner +6 margin credit) | M+W | High | Low | Trivial change to Elo update: add +6 to the winner's score before updating ratings; COOPER reports ~1% gain in win-prediction accuracy vs margin-only Elo; tests whether outcome credit on top of raw margin improves tournament feature quality; implement as an Elo variant alongside the current baseline |
| 27 | COOPER-ARCH-04 | Variable k-factor with early-season decay | M+W | Medium | Low | Use 2× base k-factor for a team's first ~15 games of each season, decaying to base thereafter; early games carry disproportionate information against a crude preseason prior; can be combined in the same run as COOPER-ARCH-01 |
| 28 | COOPER-ARCH-03 | Conference mean Elo reversion at season end | M+W | Medium | Low | Revert each team's Elo toward its conference mean (not global mean) between seasons; provides a better-grounded prior for teams in stronger conferences; requires `MTeamConferences.csv` / `WTeamConferences.csv` (confirmed present in `data/raw/march-machine-learning-mania-2026/`) |
| 29 | COOPER-FEAT-01 | Pace feature as standalone GBT input | M+W | Medium | Low | `low_pace` / `high_pace` = OT-normalized avg total game score per regular-season game per team; `pace_diff = low_pace − high_pace`; higher-scoring environments have higher score variance; GBT may learn non-linear interactions between pace and margin/efficiency features; requires compact results only |
| 30 | COOPER-ARCH-02 | Impact-factor weighted Elo / GLM fitting | M+W | Medium | Medium | Down-weight lopsided expected matchups in Elo and GLM (OLS) fitting, inversely proportional to projected score gap; conference games and tournament games receive additional weight; addresses garbage-time compression bias in blowout games; attempt only if COOPER-ARCH-01/-04 show signal |

#### Dataset-to-challenger mapping

| Dataset | Most useful challenger tasks | Recommended first use |
|---|---|---|
| `data/reference/` | `LATE-EXT-01`, `LATE-EXT-02`, routed-model diagnostics, benchmark-cell analysis | Compare local frozen predictions vs benchmark probabilities on the same all-pairs universe, then prioritize the largest recurring men/women round-bucket gaps. Promotion decisions still depend only on played-game held-out flat Brier. |
| `data/processed/betexplorer/` | `LATE-MKT-01`, later-stage calibration challengers, odds-implied strength priors | Convert opening/closing odds to implied win probabilities and test them first as a simple prior/blend ingredient before building broader market-feature bundles |
| `data/processed/espn/` | `LATE-RATE-01`, `LATE-RATE-02`, `LATE-EXT-03`, embedding challengers | Build better team-level possessions, four factors, and player-agnostic form summaries rather than immediately training a large model directly on event-level data |
| `data/processed/tourney_forecasts/` | `LATE-EXT-04` (sub-approach A first, then B and C) | Confirm data coverage for all tournament teams M+W; map team names to Kaggle TeamIDs via `data/TeamSpellings.csv`; use only forecasts from the earliest available pre-tournament date (before Selection Sunday); attempt direct `goto_conversion` H2H blend before building conditional-strength feature pipeline |

#### Recommended execution order while time remains

1. `LATE-ARCH-RG-07` — routed men `R1` / `R2+` model
2. `LATE-ARCH-RG-08` — routed women `R1` / `R2+` model
3. `LATE-RATE-01` — improved men latent strength model
4. `LATE-RATE-02` — improved women latent strength model
5. `LATE-MKT-01` — BetExplorer odds prior / calibration challenger if joins are clean
   - `LATE-EXT-04` — tournament-progression forecast challenger **if and only if** a clean 2026 data source is confirmed post-Selection Sunday; attempt sub-approach A (direct blend via `goto_conversion`) first before building conditional-strength features
6. `LATE-EXT-01` / `LATE-EXT-02` — benchmark-guided men/women gap analysis and targeted challenger design
7. `LATE-EXT-03` — ESPN-derived advanced strength/situational features
8. `LATE-EMB-01` / `LATE-EMB-02` only if the earlier, simpler challenger classes stall
9. `LATE-FEAT-21` — tournament-only Elo (low effort, new persistent-pedigree signal)
10. `LATE-FEAT-22` + `LATE-FEAT-26` + `LATE-FEAT-23` — Elo momentum, Pythagorean expectancy, seed-Elo gap (bundle: all derived from existing Elo/seed/score data)
11. `LATE-ARCH-DW-01` — decay-weighted training test on current best model per league
12. `LATE-ARCH-META-01` — logit-Ridge meta-learner swap in COMBO-03/04
13. `LATE-FEAT-25` + `LATE-FEAT-28` — season momentum and win quality bins
14. `LATE-FEAT-30` — Massey PCA and disagreement features (men only)
15. `LATE-FEAT-24` + `LATE-FEAT-27` + `LATE-FEAT-29` + `LATE-FEAT-31` — late-5 form, neutral-site profiles, conference percentile, program pedigree (if time allows)
16. `LATE-ARCH-CB-01` — CatBoost base learner (only if diversity audit shows high OOF correlation among current ensemble members)
17. `COOPER-ARCH-01` + `COOPER-ARCH-04` — win-bonus Elo and variable k-factor (trivial, implement as a single Elo variant run; test `elo_diff` from this variant as a drop-in replacement in existing models)
18. `COOPER-ARCH-03` — conference mean Elo reversion (low effort; implement as an Elo initialization change; `MTeamConferences.csv` / `WTeamConferences.csv` confirmed in `data/raw/`)
19. `COOPER-FEAT-01` — pace feature as standalone GBT input (requires only compact results; derived from existing data)
20. `COOPER-ARCH-02` — impact-factor weighted Elo / GLM fitting (medium effort; attempt only if items 17–19 show signal)

**Execution note:** `LATE-EXT-01` and `LATE-EXT-02` are benchmark-guided challenger-design tasks, not standalone promotion criteria. They exist to identify high-leverage cells and define the next testable challenger; they do not replace played-game held-out flat-Brier model selection.

#### Stop conditions

- Stop a challenger branch after two consecutive losing variants against the same frozen leader unless a new dataset or materially different modeling class is introduced.
- Do not promote a challenger on narrative appeal, apparent bracket realism, or benchmark overlap alone.
- If a challenger improves only bucket-specific diagnostics but not overall flat Brier, keep it as analysis, not as the submission-default model.

### 4.6 Architecture Experiment Gates

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

**Regression-target models (margin / point-differential output):** When the base model outputs a continuous margin or score-differential prediction rather than a probability, use a `UnivariateSpline(k=5)` fit to OOF `(margin_pred, binary_outcome)` pairs as the calibration bridge:

```python
from scipy.interpolate import UnivariateSpline
t = 25  # clip threshold; adjust if empirical range differs
spline = UnivariateSpline(
    np.clip(oof_margin_preds, -t, t),
    oof_binary_outcomes,
    k=5,
)
prob = np.clip(spline(np.clip(margin_pred, -t, t)), 0.025, 0.975)
```

The inner clip (`[-t, t]`) prevents wild extrapolation beyond the training range. `t = 25` is the default based on historical college basketball tournament margin distributions; validate that this threshold covers ≥ 99% of OOF margin observations before use, and log `spline_clip_t` as an MLflow param if it differs from 25. The outer clip ensures the output is a valid probability. The spline must always be fit on OOF predictions — never on all training data. Validate the spline fit against empirical binned win rates (bin by 2-point margin intervals) before using it for submission; acceptance criterion is ECE < 0.02 on OOF predictions, identical to isotonic regression (see calibration validation below).

**Calibration method selection rule:**
- Classification output (probability) → isotonic regression or Platt scaling
- Regression output (margin / score) → UnivariateSpline(k=5) with margin clipping
- Tournament-progression probability input (LATE-EXT-04 only) → `goto_conversion` (see below)

**Score distribution tails.** Empirically, college basketball game scores follow a slightly fat-tailed distribution — outlier outcomes (travel disruption, unusual matchup dynamics, injury during game) occur more often than a normal distribution predicts. The `UnivariateSpline` calibration handles this non-parametrically without assuming a distribution family, which is the correct approach for regression-target models. For classification-output models calibrated with isotonic regression, this is handled automatically via the empirical reliability diagram. However, if reliability diagrams show *systematic* miscalibration in the extreme bins (> 0.80 or < 0.20) after isotonic regression — not just noise — this is a signal that the base model is being fit under a normality assumption that does not hold, and a logistic or Student-t score distribution should be investigated as an alternative base.

**Favourite-longshot bias diagnostic.**

A named calibration failure mode distinct from the general overconfidence trap (§1.1): naive probability normalization systematically *underestimates strong favorites and overestimates moderate underdogs* across all games. This happens because:

1. Regular-season ratings accumulate "garbage time" where starters rest and opponents score freely, compressing ratings toward parity.
2. Tournament play has equal motivation throughout; the best teams' true advantage is larger than regular-season ratings suggest.
3. The result: a model trained on regular-season-derived features will systematically assign too much probability mass to the 0.40–0.60 range and too little to the tails (favorites > 0.75, underdogs < 0.25).

This is different from overconfidence: the concern is calibration being *too conservative on favorites*, not too aggressive. A well-functioning isotonic regression trained on sufficient historical tournament games should correct this automatically, but it must be verified explicitly.

**Favourite-longshot bias check (add to VAL-02 calibration audit):**
- After fitting calibration, plot the reliability diagram with at least 10 bins
- Check specifically whether bins above 0.70 (favorites) show systematic *under*-prediction (empirical win rate > predicted probability)
- If the calibration curve bends *below* the diagonal for favorites, the calibration is not fully correcting for this bias; consider a more flexible calibration or verify that the training window includes sufficient high-confidence games

**goto_conversion for tournament-progression probability inputs (LATE-EXT-04 only).**

When the input to H2H probability conversion is a pair of tournament round-reaching probabilities (from an external bracket forecast), use the `goto_conversion` algorithm instead of simple multiplicative normalization:

```python
# Simple multiplicative (naive — underestimates favorites):
p_low = rd_r_win_low / (rd_r_win_low + rd_r_win_high)

# goto_conversion (corrects favourite-longshot bias):
from goto_conversion import goto_conversion
odds = [1 / rd_r_win_low, 1 / rd_r_win_high]
p_low, p_high = goto_conversion(odds, multiplicativeIfImprudentOdds=True)
```

**Dependency note:** `goto_conversion` (PyPI: `goto-conversion`, v3.0.3) is not a default project dependency. Add it to `pyproject.toml` only if LATE-EXT-04 sub-approach A is promoted post-Selection Sunday.

This applies only when the base signal is tournament-progression probabilities (LATE-EXT-04). For all other calibration inputs (model probabilities or score margins), use the methods above.

**Calibration validation:**
- Reliability diagram for each league (M and W separately)
- Expected Calibration Error (ECE) < 0.02 target
- Calibrated output must be re-clipped to [0.025, 0.975] after calibration
- Favourite-longshot bias check: verify bins > 0.70 are not systematically under-predicted (see above)

### 6.2 Ensemble Construction

**Ensemble tiers (in order of complexity):**

```
Tier 0 — LOSO-fold averaging (within a single model family; CV use only)
  - When LOSO CV produces N held-out-fold models (one per season), average their
    RAW (uncalibrated) predictions across folds before feeding into Tier 1.
  - Correct order: average raw outputs → then calibrate the average once.
    Do NOT calibrate each fold independently before averaging; that would
    compound calibration artefacts and conflict with the single-calibration
    philosophy of Tier 3. This matches the modeh7 reference implementation,
    which averages raw margin predictions across fold models and calibrates once.
  - Each fold model was trained on a slightly different data subset; averaging
    reduces within-family variance at near-zero marginal cost once CV models exist.
  - CV use only: Tier 0 fold-averaging applies within cross-validation to assess
    stability. At inference time (2026 submission), there are no held-out folds
    to replicate; the selected Tier 1–3 ensemble is applied directly. Do not
    attempt to re-run all 20+ LOSO fold models on 2026 submission data.
  - This does NOT substitute for Tier 1/2 diversity across model families; it is
    complementary: Tier 0 reduces within-family variance, Tiers 1/2 reduce
    between-family variance.

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

**Margin scaling knob (for regression-target models):** Before applying spline calibration, the predicted margin can be multiplied by a scalar `α`:

```python
calibrated_prob = spline(np.clip(margin_pred * alpha, -t, t))
```

- `α > 1.0`: amplifies predictions → more extreme probabilities (aggressive)
- `α < 1.0`: dampens predictions → closer to 0.5 (conservative)
- `α = 1.0`: no modification (default; first-place solution used this)

The optimal `α` should be searched on CV, not set by intuition. Log `alpha` as an MLflow param for any regression-target model run. Do not tune `α` on the 2025 holdout.

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
| Stage 2 feature build | 2026-03-15–16 | Compute all 2026 regular-season and matchup features required by the selected frozen model family. At minimum this includes Phase A/B inputs; if the chosen model depends on Phase C or D features, build those exact inference-time features as well. If the chosen model depends on external data (for example BetExplorer odds or ESPN-derived features), validate 2026 availability, team mapping, and join completeness before the freeze is considered operational. |
| Stage 2 prep | 2026-03-16–18 | Join 2026 features with 2026 seeds; derive `round_group` and any other bracket-derived inference features required by the selected model; apply the frozen pre-fitted model to Stage 2 matchup pairs. External-data-dependent models must pass a live availability and join audit before they are eligible for submission use. |
| Final submission | 2026-03-18–19 | Upload best model output; select it explicitly before deadline |
| Hard deadline | 2026-03-19 16:00 UTC | Submission must be selected in Kaggle UI |

**Pre-Selection Sunday work (2026-03-10 to 2026-03-15):** Complete Gates 0–2 entirely. All feature engineering, model training, calibration, and ensemble construction uses historical data. The Stage 2 run is a final application of the already-selected model — not a new training run. The only allowed Stage 2 changes are inference-time feature generation for 2026 data and bracket-derived routing fields required by the chosen frozen model family.

**External-data guardrail:** no BetExplorer- or ESPN-dependent challenger may be promoted to the frozen pair unless:
- 2026 data is available on the required timeline,
- team/entity mapping is validated for the 2026 tournament field,
- join coverage is high enough that the model does not silently fall back on a minority of teams or games.

### 8.2 Submission Types

| Type | When | Description |
|---|---|---|
| Validation run | Pre-Selection Sunday | Offline scoring on 2023–2025 held-out seasons to select model |
| Stage 2 submission | Post-Selection Sunday | Apply selected model to Stage 2 pairs; upload once |
| Final selection | 2026-03-19 | Confirm submission is selected in Kaggle UI |

### 8.2.1 Gate 3 Freeze

**Frozen model choices entering Gate 3 (updated 2026-03-14):**

- Men: generalization-tuned reference-style margin model
  - MLflow run: `337d3b992b884dbb800c078561d37622`
  - 2023-2024 flat Brier: `0.195566`
  - 2025 holdout run: `a11c60d33cde4fd68f7852fc65dda1db`

- Women: `COOPER-ARCH-01 + COOPER-ARCH-04 v1` routed women + ESPN four-factor + conference-rank model with COOPER Elo replacement
  - MLflow run name: `cooper-arch-01-04-women`
  - 2023-2024 flat Brier: `0.129201`
  - 2025 holdout run name: `val-01-2025-holdout-women-cooper-arch-01-04`

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
1. First fall back to the frozen leaders already selected in this plan:
   - men: generalization-tuned reference-style margin model
   - women: `COOPER-ARCH-01 + COOPER-ARCH-04 v1`
2. If the frozen-leader path itself is unavailable operationally, fall back to the emergency minimum viable baselines:
   - men: seed-diff logistic baseline
   - women: seed-plus-Elo logistic baseline
3. Do not attempt untested code changes under deadline pressure
4. The emergency baselines are the guaranteed floor; any pipeline complexity beyond that is optional

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
| LATE-ARCH-RG-07 | Routed round-group model (M) | Separate routed models | Current frozen men feature family | M | A true routed `R1` vs `R2+` men model improves overall flat Brier where calibration-only adjustments did not | P1 |
| LATE-ARCH-RG-08 | Routed round-group model (W) | Separate routed models | Current frozen women feature family (COOPER-ARCH-01 + COOPER-ARCH-04 v1 feature set); add one women-specific feature only if needed | W | Women may benefit from stage-specific coefficient weighting even if unified calibration looks strong | P1 |
| LATE-RATE-01 | Improved latent strength model (M) | Rating model + simple downstream classifier | ESPN- and detailed-results-derived team strength features | M | A stronger upstream men strength rating improves tournament probabilities more than another broad classifier | P1 |
| LATE-RATE-02 | Improved latent strength model (W) | Rating model + simple downstream classifier | ESPN- and detailed-results-derived team strength features | W | Women still have room for improvement through upstream strength estimation rather than calibration tweaks | P1 |
| LATE-EMB-01 | Team-embedding challenger (M) | Representation learner + simple classifier | Regular-season game graph embeddings + frozen men features | M | Learned team embeddings capture matchup structure missed by scalar ratings | P3 |
| LATE-EMB-02 | Team-embedding challenger (W) | Representation learner + simple classifier | Regular-season game graph embeddings + frozen women features | W | Same, women's data | P3 |
| LATE-NN-01 | End-to-end shallow neural model | Shallow MLP | Best available late-challenger feature set | M+W | A carefully regularized shallow NN improves on linear models without calibration collapse | P4 |
| LATE-ARCH-DW-01 | Decay-weighted training (M+W) | Any base learner with exponential sample weights | Current best feature family per league | M+W | Recency-weighted training (§2.1 notes this as a possibility but no experiment exists) reduces the influence of older seasons and may improve generalization; decay tested at 0.9 (men) and 0.93 (women) | P2 |
| LATE-ARCH-META-01 | Logit-Ridge meta-learner (M+W) | Ridge regression in logit space on OOF base learner probabilities | OOF logits from current best ensemble members | M+W | Ridge in log-odds space emphasizes tail calibration where Brier loss is steepest; alpha tuned via leave-seasons-out CV; alternative to no-intercept logistic in COMBO-03/04 Tier 2 | P2 |
| LATE-ARCH-CB-01 | CatBoost base learner (M+W) | CatBoost | Current best feature family per league | M+W | Ordered boosting and categorical handling adds diverse base learner predictions; reference notebook shows CatBoost as highest meta-weight ensemble member (0.359 for men); warranted only if diversity audit shows high OOF correlation among current members | P3 |
| LATE-ARCH-MW-01 | Unified M+W model with gender flag | XGBoost/LightGBM | All Phase A+B features + `men_women` binary flag (1=M, 0=W); nullable men-only features (Massey, coach) set to 0 or league-median for women rows | M+W | Pooling men's and women's tournament data roughly doubles the training sample; XGBoost learns gender-specific patterns via the `men_women` flag. **Promotion requires all three**: (1) combined M+W flat Brier beats COMBO-05 on 2023–2024 CV; (2) men's Brier ≤ men frozen leader (0.195566); (3) women's Brier ≤ women frozen leader (0.129201). This is a challenger to committed ADR 0004 — promotion also requires an explicit update to `docs/decisions/0004-men-women-tournament-modeling-strategy.md`. | P3 |
| LATE-EXT-04 | Tournament-progression forecast challenger (M+W) | Three sub-approaches — (A) direct H2H blend via `goto_conversion`; (B) conditional-strength features as GBT inputs; (C) calibration anchor / shrinkage prior | External bracket-simulation probabilities (`rd1_win`–`rd6_win` per team; **preferred source: COOPER, Silver Bulletin** — injury-adjusted, bracket-simulates 'hot', blended 5/8 COOPER + 3/8 KenPom (M) / Her Hoop Stats (W); verify M+W coverage before Selection Sunday; fallback: ESPN, T-Rank, teamrankings.com; 538 no longer publishes); team-name join via `data/TeamSpellings.csv` | M+W | Bracket-aware pre-tournament progression probabilities are orthogonal to all internal Elo/GBT signals and encode seed draw, opponent difficulty, and simulation-derived team strength in a single compact vector; COOPER's 'hot' simulation (ratings update during bracket simulation so later-round probabilities reflect simulated early outcomes) makes it a higher-quality source than static pre-tournament rating systems; `goto_conversion` corrects favourite-longshot bias in the conversion to H2H probabilities. **Conditional on data availability confirmed post-Selection Sunday — skip entirely if clean M+W coverage not secured by 2026-03-16.** Attempt sub-approach A first; promote if flat Brier beats frozen leader on 2023–2024 CV. | P1 (if data secured) |
| COOPER-ARCH-01 | Win-bonus Elo variant (M+W) | Elo (modified update rule) | All Phase A+B features using `elo_diff` from win-bonus Elo | M+W | The standard Elo update uses raw scoring margin; adding +6 points to the winner's score before updating (empirically optimal in COOPER, derived from minimizing prediction error) corrects for the asymmetric information value of winning vs losing regardless of margin; COOPER reports ~1% improvement in win-prediction accuracy; test as a drop-in replacement for `elo_diff` in existing models without any other feature change | P2 |
| COOPER-ARCH-04 | Variable k-factor Elo variant (M+W) | Elo (modified k-factor schedule) | `elo_diff` from variable k-factor Elo; can be combined with COOPER-ARCH-01 in a single run | M+W | Use 2× base k-factor for each team's first ~15 games of the season, linearly decaying to the base k thereafter; early-season games reveal disproportionately more information relative to a crude preseason prior, but a too-high k mid-season causes zig-zagging; COOPER uses k=55 base (vs SBCB's k=38), suggesting our current k may be too conservative | P2 |
| COOPER-ARCH-03 | Conference mean Elo reversion (M+W) | Elo (modified season reversion) | `elo_diff` with conference-mean Elo reversion | M+W | Instead of reverting each team's Elo toward the global mean at season end, revert toward the mean of its conference; teams in stronger conferences have a better-grounded starting prior than the global average provides; conference membership from `MTeamConferences.csv` / `WTeamConferences.csv`; test as additional Elo variant alongside COOPER-ARCH-01/-04 | P3 |
| COOPER-ARCH-02 | Impact-factor weighted Elo / GLM fitting (M+W) | Weighted Elo; weighted OLS (GLM) | `elo_diff` from impact-weighted Elo; `glm_quality_diff` from impact-weighted OLS | M+W | Weight each game's Elo and GLM update inversely proportional to its projected score gap (lopsided blowouts contribute less signal due to garbage-time compression); conference games and tournament games receive an additional weight multiplier; addresses the same favourite-longshot bias noted in §6.1 at the feature-construction layer rather than the calibration layer; attempt only if COOPER-ARCH-01/-04 show Brier improvement on CV | P3 |

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
| LATE-FEAT-16 | BetExplorer opening odds prior | market_open_prob | Frozen leader challenger | Opening market odds provide a compact pregame strength prior that improves flat Brier | P1 |
| LATE-FEAT-17 | BetExplorer closing odds prior | market_close_prob | Frozen leader challenger | Closing market odds provide a stronger calibration anchor than internal ratings alone | P1 |
| LATE-FEAT-18 | ESPN-derived four-factor ratings | eFG, TOV, ORB, FTR composites | `LATE-RATE-01` / `LATE-RATE-02` | Richer team quality estimates from ESPN box scores improve latent strength modeling | P1 |
| LATE-FEAT-19 | ESPN-derived player-availability / rotation proxies | minutes continuity, top-player usage, lineup stability | `LATE-RATE-01` / `LATE-RATE-02` | Lineup continuity and high-usage player context add signal beyond team aggregates | P2 |
| LATE-FEAT-20 | Reference-model delta diagnostics | benchmark_prob_delta, benchmark_bucket_delta (diagnostic-only; not trainable features) | `LATE-EXT-01` / `LATE-EXT-02` | Systematic differences from the benchmark identify specific cells worth modeling, even if the benchmark fields themselves are not used directly for training | P2 |
| LATE-FEAT-21 | Tournament-only Elo (M+W) | tourney_elo_diff: separate Elo system fit only on past tournament games (K=32, season reversion=0.15) | Frozen leader challenger | Tournament-game Elo captures clutch/pressure performance signal orthogonal to full-season Elo; validated as additive in reference notebook | P2 |
| LATE-FEAT-22 | Elo momentum (M+W) | elo_momentum_diff: change in Elo from mid-season (DayNum 115) to end of regular season | Frozen leader challenger | Teams trending up entering the tournament outperform static end-of-season Elo; momentum computed from DayNum-115 Elo delta; DayNum < 134 cutoff is respected | P2 |
| LATE-FEAT-23 | Seed-Elo gap (M+W) | seed_elo_gap_diff: actual Elo minus expected Elo for that seed line | Frozen leader challenger | Negative gap (team over-seeded) signals regression risk; positive gap (team under-seeded) signals upset potential; collinear with seed_diff and elo_diff individually but their interaction is unique | P2 |
| LATE-FEAT-24 | Late-season form O/D split (M+W) | late_form_off_diff, late_form_def_diff: last-5-game offensive and defensive efficiency separately (not combined net) | Frozen leader challenger | Splitting form into offensive and defensive components captures teams riding hot offense into March versus teams winning on defense; correlates differently with upset outcomes than combined net efficiency | P2 |
| LATE-FEAT-25 | Season momentum net efficiency (M+W) | season_momentum_diff: second-half net rating minus first-half net rating (split at midseason) | Frozen leader challenger | Teams improving across the season outperform flat or declining teams at tournament time; computed from regular-season splits before DayNum 134 cutoff | P3 |
| LATE-FEAT-26 | Pythagorean expectancy (M+W) | pythag_diff: (pts_for^10.25) / (pts_for^10.25 + pts_against^10.25) differential | Frozen leader challenger | Pythagorean win expectancy (exponent 10.25) provides a more stable strength estimate than W% alone; additive over seed and Elo in reference notebook | P2 |
| LATE-FEAT-27 | Road/neutral-site performance profiles (M+W) | road_win_pct_diff, neutral_net_eff_diff | Frozen leader challenger | Tournament games are neutral-site; teams with higher neutral-site net efficiency or road win rates better approximate tournament conditions than home-inflated ratings | P2 |
| LATE-FEAT-28 | Win quality bins (M+W) | close_win_pct_diff (margin ≤5), blowout_win_pct_diff (margin ≥15) | Frozen leader challenger | Close-win rate signals resilience in tight games; blowout rate signals dominance; both are predictive in rounds where seed lines are tight — requires DayNum < 134 for cutoff compliance | P3 |
| LATE-FEAT-29 | Conference percentile rank (M+W) | conf_pct_rank_diff: team's within-conference rank percentile | Frozen leader challenger | Conference percentile rank normalizes for conference depth and correlates with tournament seeding committee adjustments; lower collinearity with raw Elo than raw conference rank | P3 |
| LATE-FEAT-30 | Massey PCA + disagreement (M) | massey_pca1_diff, massey_disagreement_diff: first PC of Massey system ratings and cross-system std | Frozen leader challenger (men only; women Massey data is sparse) | First PC captures consensus latent strength across 100+ systems; disagreement std captures uncertainty orthogonal to consensus; complement FEAT-04 single-system rank | P2 |
| LATE-FEAT-31 | Tournament program pedigree (M+W) | pedigree_score_diff: 5-year tournament win rate weighted by round reached | Frozen leader challenger | Program pedigree captures institutional tournament experience beyond coaching tenure; 5-year window balances recency and sample size; computed from historical tournament results (no leakage by construction) | P3 |
| MH7-FEAT-01 | GLM team quality (Bradley-Terry OLS) (M+W) | `glm_quality_diff = α_LowTeamID - α_HighTeamID` (positive when LowTeam is stronger); OLS fit per season independently on that season's regular-season games only (no cross-season pooling; complies with §2.2 leakage policy); design matrix: one row per game, one column per team, +1 for the winning team, −1 for the losing team, no intercept; target: `WScore - LScore` (winner's margin); estimated via `statsmodels.OLS(y, X).fit()` where `X` is the sparse indicator matrix and `y` is the point differential vector; coefficients α_i are in points | Frozen leader challenger | Simultaneous estimation of all team strengths in one season-wide OLS pass captures the full game covariance structure, orthogonal to both sequential Elo and iterative SOS; modeh7 first-place solution shows AUC ~0.84 standalone, additive over seeds and Elo as a GBT feature | P2 |
| MH7-FEAT-02 | Opponent raw box-score averages (M+W) | Six stats, each as `low_avg_opp_{stat}`, `high_avg_opp_{stat}`, `avg_opp_{stat}_diff` (Low − High), where `{stat}` ∈ {Score, FGA, Blk, PF, TO, Stl}: for each team, the average stats recorded **by opponents against that team** across regular-season games (OT-normalized per §3.3). `low_avg_opp_Score` = mean points allowed by LowTeamID per game; higher values indicate weaker defense. All six diffs follow the schema convention (LowTeamID value minus HighTeamID value). Men's detailed results available from 2003; women's from 2010; training floor is 2010 for both leagues per §2.4 — pre-2010 rows have these columns set to null | Frozen leader challenger | Raw opponent-perspective counting stats provide a simpler, interpretable defensive quality and schedule-strength proxy that is orthogonal to possession-adjusted efficiency; GBT can learn non-linear combinations that complement `def_eff` | P3 |
| GC-FEAT-01 | Conditional round-progression strength (M+W) | `low_cond_r{n}_strength = rd{n}_win / rd{n-1}_win` for n ∈ {2, 3, 4} per team from external bracket forecast; `cond_r{n}_strength_diff = low_cond_r{n}_strength − high_cond_r{n}_strength` (positive when LowTeam has higher conditional strength at round n); use only forecasts from the earliest available pre-tournament date (before Selection Sunday) | `LATE-EXT-04` (sub-approach B); requires `data/processed/tourney_forecasts/` with M+W coverage confirmed post-Selection Sunday | Conditional round-progression probabilities extract bracket-position-aware team strength at each round independently of the R1 matchup; a team with high `cond_r2_strength` is likely to win R2 even after controlling for their R1 seed, providing signal orthogonal to Elo and seed features; join to matchup table via `data/TeamSpellings.csv` | P1 (if LATE-EXT-04 data secured) |
| COOPER-FEAT-01 | Pace feature (M+W) | `low_pace` = OT-normalized avg of (LowTeamID score + opponent score) per regular-season game; `high_pace` equivalent; `pace_diff = low_pace − high_pace` (positive when LowTeam plays in higher-scoring environments) | Frozen leader challenger (GBT; add to current best feature set per league) | Higher-scoring game environments have higher score variance (a point is worth less signal when 160 total points are scored vs 120); GBT can learn non-linear interactions between pace and margin/efficiency features that scalar ratings miss; also serves as a normalization baseline for `avg_opp_Score` (MH7-FEAT-02), distinguishing teams that allow many points because they face high-pace opponents from teams with genuinely weak defense; derived from compact results only — no detailed-results dependency | P3 |

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
| LATE-VAL-06 | Market-data coverage and leakage audit | BetExplorer joins by season/league | Join-rate table + held-out audit | Market odds can be used without silent team-mapping leakage or severe season gaps | P1 |
| LATE-VAL-07 | ESPN feature stability audit | ESPN-derived team features | Season-by-season null/coverage profile | ESPN-derived features are stable enough across seasons to support a deadline-safe challenger | P1 |
| LATE-VAL-08 | Benchmark all-pairs gap analysis | Local frozen leaders vs `data/reference/` | All-matchups comparison by round/bucket | Benchmark comparisons identify a small number of recurring, high-leverage cells worth targeted challenger work | P1 |

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
- [ ] Men frozen leader: generalization-tuned reference-style margin model
- [ ] Women frozen leader: `COOPER-ARCH-01 + COOPER-ARCH-04 v1` routed women + ESPN four-factor + conference-rank model with COOPER Elo replacement
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
