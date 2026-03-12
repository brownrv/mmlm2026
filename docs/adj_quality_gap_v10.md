# adj_quality_gap_v10 — Model Documentation

**Competition:** Kaggle March Machine Learning Mania 2026
**Metric:** Brier score (lower is better); equivalent to mean squared error on probabilities
**Task:** Predict P(TeamID1 wins) for every possible men's and women's NCAA tournament matchup
**Source file:** `src/kaggle_mmlm/models/adj_quality_gap_v10.py`

---

## 1. Overview

`adj_quality_gap_v10` is a two-league (men / women) ensemble model that combines:

1. A **KenPom-style SOS-adjusted efficiency model** — iterative strength-of-schedule adjustment of per-game offensive and defensive efficiency ratings derived from box scores
2. An **Elo rating system** — time-series rating model updated after every regular-season and tournament game
3. A **linear blend** of the two probabilities with a league-specific weight `alpha`

Men and women use meaningfully different sub-architectures. A brief comparison:

| Component | Men | Women |
|-----------|-----|-------|
| Base learner | HistGradientBoostingRegressor on game margin | Logistic Regression on win/loss |
| Features | 8 features | 3 features |
| SOS HCA correction | No | Yes (HCA = 3.0 pts) |
| Temperature scaling | Dynamic, shrinkage-blended | T = 1.0 (none) |
| Elo alpha | Walk-forward CV over last 8 seasons | Fixed at 0.10 |
| Close-game threshold | CloseWR1 (margin ≤ 1 pt) | CloseWR3 (margin ≤ 3 pts) |

---

## 2. Data Sources

All data comes from the Kaggle competition dataset. No external data is used.

| File | Usage |
|------|-------|
| `{M\|W}RegularSeasonDetailedResults.csv` | Efficiency, MOV, assist rate, close-game WR, FTR, TOVr |
| `{M\|W}RegularSeasonCompactResults.csv` | Elo rating inputs (via `load_games`) |
| `{M\|W}NCAATourneyCompactResults.csv` | Historical tourney outcomes for training; also added to Elo |
| `{M\|W}NCAATourneySeeds.csv` | SeedDiff feature; used to align canonical matchup rows |
| `{M\|W}ConferenceTourneyGames.csv` | Conference tournament win rate (men only) |

---

## 3. Preprocessing

### 3.1 Possession Estimate (Dean Oliver)

Applied to the detailed results to convert raw scores into efficiency ratings:

```
Poss_W = WFGA - WOR + WTO + 0.44 * WFTA
Poss_L = LFGA - LOR + LTO + 0.44 * LFTA
Poss   = (Poss_W + Poss_L) / 2          # average per game

W_OffEff = WScore / Poss * 100          # offensive efficiency per 100 poss
W_DefEff = LScore / Poss * 100          # defensive efficiency per 100 poss
L_OffEff = LScore / Poss * 100
L_DefEff = WScore / Poss * 100
MOV_per100 = (WScore - LScore) / Poss * 100
```

### 3.2 Home Court Adjustment (Women Only, HCA = 3.0)

Before running the SOS iteration, efficiency figures are corrected for home/away venue bias. This prevents home-game advantages from inflating a team's apparent quality. The adjustment is applied to the **raw scores** (not the efficiency values directly) before recomputing efficiency:

```
h2 = HCA / 2  = 1.5 raw points

If WTeam is home (WLoc == 'H'):
    adj_w = -h2   # remove home scoring inflation from WTeam
    adj_l = +h2   # remove away scoring suppression from LTeam

If WTeam is away (WLoc == 'A'):
    adj_w = +h2   # restore away scoring suppression for WTeam
    adj_l = -h2

Neutral (WLoc == 'N'):
    adj_w = adj_l = 0

Then recompute efficiency from adjusted scores:
    W_OffEff = (WScore + adj_w) / Poss * 100
    W_DefEff = (LScore + adj_l) / Poss * 100
    L_OffEff = (LScore + adj_l) / Poss * 100
    L_DefEff = (WScore + adj_w) / Poss * 100
```

Note: because the adjustment is applied to the raw score before dividing by possessions, the effective efficiency adjustment is `h2 / Poss * 100`. For a typical 70-possession game this is approximately 2.1 efficiency units, not 1.5.

This step is applied **only for the women's league** and **only for the SOS iteration input** (§3.3). All other derived statistics (MOV, CloseWR3, FTR, TOVr, AstRate) continue to use the raw unadjusted scores.

### 3.3 SOS-Adjusted Efficiency (KenPom-style Iterative Convergence)

Applied **per season** independently, using **regular-season games only** (tournament games are excluded). The goal is to produce adjusted offensive (AdjO) and defensive (AdjD) ratings that account for opponent quality.

**Initialization:**
```
AdjO[team] = mean(OffG across all games)
AdjD[team] = mean(DefG across all games)
```

**Iteration** (up to `n_iter=100` times, converges when mean |ΔAdjO| < `tol=1e-4`):
```
At the start of each iteration, compute season-level league averages
from the CURRENT AdjO/AdjD values:
    LeagueAvgO = mean(AdjO) across all teams in that season
    LeagueAvgD = mean(AdjD) across all teams in that season

For each game, compute opponent-adjusted game efficiency:
    adjO_game = OffG / OppAdjD * LeagueAvgD
    adjD_game = DefG / OppAdjO * LeagueAvgO

Then update:
    AdjO[team] = mean(adjO_game across all team's games)
    AdjD[team] = mean(adjD_game across all team's games)
```

LeagueAvgO and LeagueAvgD are recomputed from the current AdjO/AdjD at the start of each iteration (not fixed at initialization values).

**Output:**
```
AdjNetEff = AdjO - AdjD
```

The women's computation uses HCA-corrected efficiency inputs; the men's uses raw inputs.

---

## 4. Feature Engineering

All features are expressed as **differentials** (Team A value minus Team B value) where Team A is defined by the canonical ordering (lower TeamID). This enforces antisymmetry: swapping the teams flips all feature signs and flips the outcome, which is enforced explicitly during training (see §6.1).

### 4.1 AdjQG — Adjusted Quality Gap

```
AdjQG = AdjNetEff_A - AdjNetEff_B
```

The main strength-of-schedule-adjusted efficiency differential. Highly correlated with KenPom's NetRtg (r ≈ 0.991 across seasons).

### 4.2 d_MOV — Margin of Victory per 100 Possessions

```
TeamMOV = mean(MOV_per100) across all regular-season games
d_MOV   = TeamMOV_A - TeamMOV_B
```

Uses raw (non-SOS-adjusted) scoring margins normalized by possessions. Captures consistent dominance independent of opponent quality. Correlated with AdjQG (r ≈ 0.74) but retains independent predictive signal; removing d_MOV increases Brier by ~+0.00038.

### 4.3 SeedDiff — Tournament Seed Differential

```
SeedValue = integer seed number (1–16)

Extraction from raw seed string (e.g. "W01", "X12", "Y16b"):
    SeedValue = first integer found by regex r"\d+" in the seed string
    (strips leading region letter W/X/Y/Z and trailing play-in suffix a/b)
    Default: 16 if extraction fails

SeedDiff  = SeedValue_A - SeedValue_B
```

Only applicable after Selection Sunday. For pre-tournament predictions, seeds are unavailable and this feature is zeroed out (missing → 0 imputation). At training time all games have seeds since they are historical tournament games.

### 4.4 d_AstRate — Assist Rate Differential (Men only)

```
AstRate = sum(Ast) / sum(FGM)  per team-season
d_AstRate = AstRate_A - AstRate_B
```

A proxy for ball movement quality and team cohesion.

### 4.5 d_CloseWR1 — Close-Game Win Rate Differential (Men only, ≤ 1-pt margin)

```
CloseWR1 = wins in games with |WScore - LScore| <= 1
           / total games with |WScore - LScore| <= 1
d_CloseWR1 = CloseWR1_A - CloseWR1_B
```

Captures clutch performance and composure in ultra-tight games. Uses a 1-point threshold (stricter than the 3-point version used in older models). Teams with no close games receive CloseWR1 = 0 (zero imputation). Experiment item 12: this threshold improved over CloseWR3 (Δ = −0.00133 Brier, 10/15 seasons improved).

### 4.6 d_CloseWR3 — Close-Game Win Rate Differential (Women only, ≤ 3-pt margin)

```
CloseWR3 = wins in games with |WScore - LScore| <= 3
           / total games with |WScore - LScore| <= 3
d_CloseWR3 = CloseWR3_A - CloseWR3_B
```

Same concept as d_CloseWR1 but with a 3-point threshold. Used in the women's model where the 1-point threshold produces too few samples per team per season.

### 4.7 d_ConfTourneyWR — Conference Tournament Win Rate Differential (Men only)

```
ConfTourneyWR = wins in conference tournament games
                / total conference tournament games played
d_ConfTourneyWR = ConfTourneyWR_A - ConfTourneyWR_B
```

Source: `MConferenceTourneyGames.csv`. Teams that did not participate in a conference tournament (e.g., Ivy League auto-bid) receive ConfTourneyWR = 0 (zero imputation; league-mean imputation was tested and hurt).

### 4.8 d_FTR — Free Throw Rate Differential (Men only)

```
FTR = sum(FTM) / sum(FGA)  per team-season (across all regular-season games)
d_FTR = FTR_A - FTR_B
```

Uses FTM (free throws made) divided by FGA (field goal attempts), Dean Oliver's preferred formulation. The FTA/FGA variant was tested and lost by +0.00243 Brier.

### 4.9 d_TOVr — Turnover Rate Differential (Men only)

```
TOVr = sum(TO) / Poss_team   per team-season (season aggregate)
       where Poss_team = sum(FGA) - sum(OR) + sum(TO) + 0.44 * sum(FTA)
             and all sums use the team's own offensive stats
d_TOVr = TOVr_A - TOVr_B
```

The possessions denominator is computed at the **season-aggregate level** using each team's own offensive rebounds (`OR = WOR` when the team won, `OR = LOR` when the team lost). This is distinct from the game-averaged `Poss` in §3.1, which averages both teams' possession estimates.

Turnover rate relative to possession count. Positive d_TOVr means Team A turns the ball over more frequently than Team B.

---

## 5. Elo Rating System

Ratings are updated chronologically game-by-game across regular season and tournament games using a Margin-of-Victory-weighted Elo formula.

### 5.1 Update Rule

```
Expected win probability for winner:
    E_W = 1 / (1 + 10^((R_L - R_W) / scale))

MOV multiplier (if mov_alpha > 0):
    mov_mult = 1 + (WScore - LScore) / mov_alpha
    (allows margin to amplify or reduce the update)

K-factor (with home adjustment):
    boost = home_advantage  if WTeam is home
    boost = -home_advantage if WTeam is away
    boost = 0               if neutral

    Adjusted expected probability:
        E_W_adj = 1 / (1 + 10^((R_L - R_W - boost) / scale))

Rating update:
    delta = weight * k_factor * mov_mult * (1 - E_W_adj)
    R_W += delta
    R_L -= delta

# Rating floor: if R_L drops below 1.0, clamp to 1.0
R_L = max(R_L, 1.0)
```

### 5.2 Season-to-Season Carryover

At the start of each new season, ratings are shrunk toward the initial rating:
```
R_new = season_carryover * R_old + (1 - season_carryover) * initial_rating
```

### 5.3 Elo Hyperparameters

| Parameter | Men | Women |
|-----------|-----|-------|
| `initial_rating` | 1618.0 | 1213.0 |
| `k_factor` | 76.0 | 136.0 |
| `home_advantage` | 43.0 | 119.0 |
| `season_carryover` | 0.9780 | 0.9840 |
| `scale` | 1835.34 | 1888.27 |
| `mov_alpha` | 6.5450 | 11.4612 |
| `weight_regular` | 1.4674 | 1.4974 |
| `weight_tourney` | 0.8204 | 0.5002 |

The `scale` parameter is substantially larger than the traditional Elo value of 400 because it was jointly optimised with `k_factor` — the ratio k/scale determines update step size, so the two parameters are not independently interpretable.

#### Tuning Procedure

Parameters were tuned using **Optuna** (Tree-structured Parzen Estimator, TPE) via `scripts/tune_elo.py`.

- **Objective metric:** Mean Brier score `(1 - exp_w)^2` over tournament games in the Stage 1 seasons (2022–2025), where `exp_w` is the **pre-game home-adjusted Elo win probability** computed before the K-factor update. Only tournament games (`tourney=1`) contribute to the loss; regular-season games are excluded.
- **Training data:** All games available (regular season + tournament) through the most recent stage-1 season; Elo is evaluated by computing ratings game-by-game and scoring against the tournament outcomes.
- **Search space:** Broad initial sweep, followed by a narrow re-tune (±25% around the initial best). Final runs used 500 trials each, seed=42.
- **Search ranges (broad):** initial_rating [800–2000], k_factor [20–200], home_advantage [30–200], season_carryover [0.5–1.0], scale [200–2000], mov_alpha [0–50], weight_regular/weight_tourney [0.5–1.5].

### 5.4 Elo Cutoff

Elo ratings used for predicting season `S` are computed on all games where:
```
Season < S  OR  (Season == S AND DayNum < 134)
```

`DayNum 134` is the first possible men's play-in game day. The same numerical cutoff is applied for both men and women (women's DayNums vary by era but regular-season games consistently end well before day 134). This ensures the model never uses tournament results to predict that same tournament — only regular-season games from the target season are included.

### 5.5 Elo Win Probability

```
p_elo = 1 / (1 + 10^((R_TeamB - R_TeamA) / scale))
```

where `scale` is the league-specific tuned value from §5.3 (1835.34 for men, 1888.27 for women).

---

## 6. Men's Model

### 6.1 Training Data Construction (Mirrored Matchups)

For each historical tournament game, two rows are created. **Play-in games (Round 0) are included in training** alongside R64–Champ games (removing them hurts Brier by +0.00093, see §13).

The first available training season for men is 2004 (men's detailed results begin 2003; at least one prior season is needed). Women's detailed results begin 2010.

**Base row:** TeamA = min(WTeamID, LTeamID), TeamB = max(WTeamID, LTeamID)
```
y = 1.0 if TeamA won else 0.0
TeamA_margin = WScore - LScore  (positive if TeamA won)
Features: [AdjQG, d_MOV, SeedDiff, d_AstRate, d_CloseWR1,
           d_ConfTourneyWR, d_FTR, d_TOVr]  (all from Team A's perspective)
```

**Mirror row:** All differential features are negated, `y = 1 - y`, margin is negated.

This mirroring trick:
- Forces the model to be antisymmetric (swapping teams exactly flips the prediction)
- Doubles the effective training set without adding new information
- Effectively pins the intercept to 0 (a 50/50 matchup → 50% probability)

Games with missing `AdjQG`, `d_MOV`, or `SeedDiff` are dropped before training. Missing values in other features are imputed with 0 at training and inference time.

Training uses all historical seasons **strictly less than** the target season (leave-future-out, a.k.a. walk-forward or expanding window).

### 6.2 HistGradientBoostingRegressor

The base learner predicts **point margin** (TeamA_margin), not win probability directly.

**Algorithm:** Histogram-based gradient boosting on squared-error loss (regression).
**Implementation:** `sklearn.ensemble.HistGradientBoostingRegressor`

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `max_iter` | 200 |
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `min_samples_leaf` | 20 |
| `random_state` | 42 |
| Loss function | `squared_error` (default) |

Tuning note: depth=4 overfits on 2025 data and was worse in LOSO (Δ = +0.00317 Brier); defaults were retained.

### 6.3 Margin to Probability Conversion

The HGB's predicted margin is converted to a win probability via the Gaussian CDF:

```
sigma = std(y_train - y_hat_train)       # training residual std dev
sigma = max(sigma, 0.5)                  # floor to prevent near-zero sigma

p_raw = Phi(predicted_margin / sigma)   # Phi = standard normal CDF
```

The residual standard deviation `sigma` represents the model's uncertainty — if the model predicted a 10-point margin and `sigma = 14`, the win probability is Phi(10/14) ≈ 0.76.

### 6.4 Temperature Scaling (Men only)

Temperature scaling adjusts calibration:
```
p_cal = sigmoid(logit(p_raw) / T)
      = 1 / (1 + exp(-logit(p_raw) / T))
```

- `T > 1` → flatter (more uncertain) predictions
- `T < 1` → sharper (more confident) predictions
- `T = 1` → identity (no change)

#### Dynamic T Estimation

The optimal T is estimated from the observation that tournament **chalkiness is predictable** from field-level efficiency dispersion:

```
eff_sigma = std(AdjNetEff) across all teams in the tournament field
```

Historical correlation: r(eff_sigma, upset_rate) ≈ −0.578 (p = 0.005, n ≈ 40 seasons). A wider field distribution → more chalk → sharper predictions appropriate (lower T).

**Walk-forward procedure** with two separate data windows:

*Per-season T (for OLS regression):* iterate over **all training seasons from index 5 onward** (`train_seasons[5:]`). For each season s:
1. Re-train a 4-feature HGB (features: AdjQG, d_MOV, SeedDiff, d_AstRate) on matchups[Season < s]
2. Compute sigma from training residuals (full mirrored training set)
3. Predict on the **first half (non-mirrored, base rows only)** of season s's matchups
4. Find optimal T_s via bounded scalar minimization of log-loss over [0.1, 5.0]
5. Compute eff_sigma_s for season s's tournament field (requires seeds, so seasons without seeds are skipped)

Skip season s entirely if held-out set has < 20 games.

Note: the T estimation uses the same 4-feature HGB as the alpha estimation (§6.5), not the full 8-feature v10 HGB. This is valid because T depends primarily on eff_sigma (field dispersion), which is independent of the feature set.

**OLS regression** (fit to all `len(season_T)` points collected above):
```
T_s ~ b0 + b1 * eff_sigma_s
T_dynamic = clip(b0 + b1 * eff_sigma_target, 0.5, 3.0)
```

**Pooled T:** Collect predicted probabilities from only the **last `CAL_SEASONS = 8` training seasons** (`idx >= max(5, len(train_seasons) - 8)`), then minimize log-loss jointly → `T_pooled`

**Fallback conditions:**
- `len(train_seasons) < 5` → return T = 1.0 immediately
- `len(season_T) < 5` OR `eff_sigma_target` is NaN (seeds not available) → return `T_pooled`

**Shrinkage blend:**
```
w = |r(eff_sigma, T)|       # Pearson correlation as blend weight, clipped to [0, 1]
T_final = w * T_dynamic + (1 - w) * T_pooled
T_final = clip(T_final, 0.5, 3.0)
```

The shrinkage blend is motivated by the moderate but noisy relationship between eff_sigma and optimal T — using eff_sigma fully would overfit on noise.

**2025 example:** eff_sigma = 15.12 (high chalk field) → T_final = 0.88 (sharper than neutral)
**2026 example:** Seeds not yet available → T = 1.17 (pooled fallback)

### 6.5 Elo Blend Alpha Estimation (Men)

Walk-forward CV over the last `CAL_SEASONS_ALPHA = 8` training seasons to estimate the optimal linear mix:

```
p_blend = alpha * p_elo + (1 - alpha) * p_cal
```

For each calibration season s:
1. Re-train a 4-feature HGB (AdjQG, d_MOV, SeedDiff, d_AstRate) on mirrored matchups[Season < s]
2. Compute p_v4 for **base (non-mirrored) rows** of season s using the HGB + sigma + T scaling
3. Compute p_elo for the same rows using the Elo ratings trained on games[Season < s OR (Season == s AND DayNum < 134)]
4. Find `alpha_s = argmin_Brier(alpha * p_elo + (1-alpha) * p_v4)` over [0, 1] via bounded scalar optimization

The alpha optimization uses **Brier score** (not log-loss) on the base (non-mirrored) canonical rows.

**Final alpha:** Mean of per-season optima: `alpha = mean(alpha_per_season)`
**Fallback:** If fewer than 5 calibration seasons available → `ALPHA_FALLBACK_MEN = 0.45`

Note: The 4-feature HGB is used (not all 8 v10 features). The alpha estimate is robust to this because it depends primarily on the relative predictive power of Elo vs. any reasonable efficiency model, which is mostly captured by AdjQG.

**2025/2026 value:** alpha ≈ 0.47–0.51 (std ≈ 0.36 across seasons)

---

## 7. Women's Model

### 7.1 Training Data Construction

Identical mirroring procedure to the men's model (§6.1), including play-in games, but using a 3-feature set and the HCA-corrected AdjNetEff. The target variable is win/loss (`y`), not game margin. The first available training season is 2011 (women's detailed results begin 2010).

### 7.2 Logistic Regression

**Algorithm:** Standard logistic regression on binary outcomes.
**Implementation:** `sklearn.linear_model.LogisticRegression`

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `C` (regularization) | 1.0 (default) |
| `max_iter` | 500 |
| `random_state` | not set (lbfgs is deterministic; result is reproducible) |
| Solver | `lbfgs` (sklearn default) |
| Penalty | `l2` (default) |

The women's model uses logistic regression rather than gradient boosting because the per-season tournament sample is small (~64 games per season), causing HGB to overfit. Women HGB was tested and lost by +0.010 Brier (experiment W1, 2–3 of 14 seasons improved).

### 7.3 Temperature Scaling (Women)

Women's temperature is fixed at **T = 1.0** (no scaling). Walk-forward calibration showed that pooled T estimation **hurts** for women (Δ = +0.00154 Brier). This was confirmed in LOSO notebook 12.

### 7.4 Elo Blend Alpha (Women)

Alpha is fixed at **0.10** (i.e., 10% Elo, 90% logistic). Walk-forward alpha estimation was tested and showed minimal improvement (experiment W2: Δ = −0.00017 Brier, 8/12 seasons), not worth the additional complexity.

---

## 8. Final Prediction Formula

### Men
```
p_raw   = Phi(HGB_predict(X_men) / sigma)
p_cal   = sigmoid(logit(p_raw) / T_final)
p_elo   = 1 / (1 + 10^((R_B - R_A) / scale_men))     # scale_men = 1835.34
Pred    = clip(alpha_men * p_elo + (1 - alpha_men) * p_cal, 1e-10, 1 - 1e-10)
```

### Women
```
p_base  = logistic_model.predict_proba([AdjQG, SeedDiff, d_CloseWR3])[:, 1]
p_elo   = 1 / (1 + 10^((R_B - R_A) / scale_women))   # scale_women = 1888.27
Pred    = clip(0.10 * p_elo + 0.90 * p_base, 1e-10, 1 - 1e-10)
```

Predictions are clipped to [1e-10, 1−1e-10] to avoid log-score singularities (though the competition uses Brier score, not log-loss).

---

## 9. Submission Format

The model outputs probabilities that **TeamID1 wins**, where TeamID1 < TeamID2 (the lower numeric ID is always first). The submission ID format is `{Season}_{TeamID1}_{TeamID2}`.

Men's TeamIDs are in the range 1000–1999; women's are 3000–3999. The model is called separately for each league and the outputs are concatenated.

---

## 10. Inference / Auxiliary Outputs

In addition to `Pred`, the model returns KenPom-style score predictions for use in the bracket dashboard (not submitted to Kaggle):

**Tempo** is the average possessions per game for a team across all regular-season games in that season, computed from `RegularSeasonDetailedResults`:
```
Tempo[team] = mean(Poss) across all team's regular-season games
            where Poss = (W_Poss + L_Poss) / 2  (per game, from §3.1)
```

**Predicted score and pace:**
```
PredPace      = (Tempo_TeamID1 + Tempo_TeamID2) / 2
PredScore_1   = AdjO_1 * (AdjD_2 / LeagueAvgD) * PredPace / 100
PredScore_2   = AdjO_2 * (AdjD_1 / LeagueAvgD) * PredPace / 100
PredMargin_KP = PredScore_1 - PredScore_2
```

where `LeagueAvgD` is the season-level mean of `AdjD` across all teams, and `AdjO_1`, `AdjD_1` etc. come from the SOS iteration (§3.3).

---

## 11. Hyperparameter Reference Table

### SOS Iteration

| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_iter` | 100 | Max iterations (converges in ~19 typically) |
| `tol` | 1e-4 | Convergence threshold on mean |ΔAdjO| |

### HGB (Men)

| Parameter | Value |
|-----------|-------|
| `max_iter` | 200 |
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `min_samples_leaf` | 20 |
| `random_state` | 42 |

### Temperature Scaling (Men)

| Parameter | Value |
|-----------|-------|
| `cal_seasons` | 8 (walk-forward window) |
| T range | [0.1, 5.0] for per-season optimization |
| T final clip | [0.5, 3.0] |

### Elo (see §5.3 for full table)

### Elo Blend Alpha

| Parameter | Men | Women |
|-----------|-----|-------|
| `CAL_SEASONS_ALPHA` | 8 | N/A (fixed) |
| `ALPHA_FALLBACK_MEN` | 0.45 | — |
| `ALPHA_FALLBACK_WOMEN` | — | 0.10 |
| alpha range | [0.0, 1.0] | fixed |

### HCA (Women SOS only)

| Parameter | Value |
|-----------|-------|
| `WOMEN_HCA` | 3.0 pts |

---

## 12. Evaluation

### Loss Function (Competition)

**Brier score:**
```
BS = (1/N) * sum((Pred_i - y_i)^2)
```
where `y_i = 1` if the first team (lower ID) won, else `y_i = 0`.

All model selection and hyperparameter decisions in this project used Brier score. Log-loss was monitored as a secondary diagnostic but never optimized.

### Cross-Validation Strategy

**Leave-Season-Out (LOSO):** Train on all seasons < S, evaluate on season S. Repeat for each target season S in the historical range. This is equivalent to an expanding-window time-series split and prevents any data leakage.

### Performance Summary

| League | Metric | Value |
|--------|--------|-------|
| Men | Stage 1 Brier (2022–2025 all matchups) | 0.15741 |
| Women | 2025 LOSO Brier (63 played games) | 0.09810 |
| Both | Kaggle leaderboard rank | 197 / 1727 (top 11%) |

Note: The Stage 1 leaderboard score (0.15741) is computed over **all possible matchups** in the submission template — the vast majority of which involve teams that never played each other. The LOSO figures below are computed only on games that were **actually played** (63 per season), so the two numbers are not directly comparable.

### LOSO Brier by Season

Leave-Season-Out expanding window (train on all < S, evaluate on S). Rounds 1–6 only (play-in excluded from the overall figures below). Women's tournament started in 1998; detailed results needed for AdjQG available from 2010.

| Season | Men Brier | Men N | Women Brier | Women N | Combined |
|--------|-----------|-------|-------------|---------|----------|
| 2010 | 0.18784 | 63 | — | — | 0.18784 |
| 2011 | 0.22443 | 63 | 0.11886 | 63 | 0.17164 |
| 2012 | 0.17633 | 63 | 0.10033 | 63 | 0.13833 |
| 2013 | 0.20746 | 63 | 0.15113 | 63 | 0.17930 |
| 2014 | 0.19308 | 63 | 0.12651 | 63 | 0.15979 |
| 2015 | 0.14932 | 63 | 0.10481 | 63 | 0.12706 |
| 2016 | 0.17952 | 63 | 0.16101 | 63 | 0.17026 |
| 2017 | 0.15893 | 63 | 0.12767 | 63 | 0.14330 |
| 2018 | 0.19830 | 63 | 0.15323 | 63 | 0.17577 |
| 2019 | 0.15325 | 63 | 0.12346 | 63 | 0.13835 |
| 2020 | — | — | — | — | (no tournament — COVID) |
| 2021 | 0.21480 | 62 | 0.14260 | 63 | 0.17842 |
| 2022 | 0.21480 | 63 | 0.14140 | 63 | 0.17810 |
| 2023 | 0.20809 | 63 | 0.16504 | 63 | 0.18657 |
| 2024 | 0.18349 | 63 | 0.11262 | 63 | 0.14806 |
| 2025 | 0.13095 | 63 | 0.09810 | 63 | 0.11452 |
| **Mean** | **0.18534** | **944** | **0.13048** | **882** | **0.15884** |
| **Stage1 (2022–2025)** | **0.18433** | **252** | **0.12929** | **252** | **0.15681** |

Year-to-year variation is primarily driven by the upset rate in a given season: high-chalk tournaments (e.g. 2025 men) produce lower Brier scores, while chaotic tournaments (e.g. 2023) produce higher scores. There is no significant era trend (see peer review item 30).

### 2025 Brier by Round

Play-in games (Round 0) are excluded from the overall figures above but shown here for completeness.

| Round | Name | Men Brier | Men N | Women Brier | Women N | Combined |
|-------|------|-----------|-------|-------------|---------|----------|
| 0 | Play-in | 0.22132 | 4 | 0.31448 | 4 | 0.26790 |
| 1 | R64 | 0.11087 | 32 | 0.08928 | 32 | 0.10007 |
| 2 | R32 | 0.18766 | 16 | 0.11051 | 16 | 0.14908 |
| 3 | S16 | 0.06211 | 8 | 0.11090 | 8 | 0.08651 |
| 4 | E8 | 0.09476 | 4 | 0.06466 | 4 | 0.07971 |
| 5 | F4 | 0.22687 | 2 | 0.09726 | 2 | 0.16206 |
| 6 | Champ | 0.36970 | 1 | 0.21475 | 1 | 0.29223 |
| **Overall** | **R1–R6** | **0.13095** | **63** | **0.09810** | **63** | **0.11452** |

### 2025 Brier by Occurrence-Probability Bucket

Occurrence-probability buckets map **purely by round** — all games within a given round are placed in the same bucket. There is no within-round distinction based on which specific teams are involved or their matchup-level occurrence probability. For example, every E8 game (regardless of seed combination) is "Plausible", because on average only ~50% of teams predicted to reach the Elite Eight actually get there.

| Bucket | Rounds | Men Brier | Men N | Women Brier | Women N | Combined |
|--------|--------|-----------|-------|-------------|---------|----------|
| Definite | R64 | 0.11087 | 32 | 0.08928 | 32 | 0.10007 |
| Very Likely | R32 | 0.18766 | 16 | 0.11051 | 16 | 0.14908 |
| Likely | S16 | 0.06211 | 8 | 0.11090 | 8 | 0.08651 |
| Plausible | E8 | 0.09476 | 4 | 0.06466 | 4 | 0.07971 |
| Remote | F4 + Champ | 0.27448 | 3 | 0.13642 | 3 | 0.20545 |
| **Overall** | **All** | **0.13095** | **63** | **0.09810** | **63** | **0.11452** |

---

## 13. Key Experimental Findings

The following design decisions were validated through controlled ablations (see `PEER_REVIEW_TRACKER.md` and notebooks 12–29):

| Decision | Experiment | Result |
|----------|-----------|--------|
| HGB over logistic for men | v3 design | HGB wins by wide margin |
| Logistic over HGB for women | W1 | HGB +0.010 worse (small N per season) |
| d_MOV retained | Item 13 | Removing hurts +0.00038 |
| CloseWR1 threshold (men) | Item 12 | Δ = −0.00133 vs CloseWR3, 10/15 seasons |
| CloseWR3 threshold (women) | Item 12 | 1-pt too sparse; 3-pt optimal |
| ConfTourneyWR zero imputation | Item 14 | League-mean imputation hurts +0.00070 |
| Women T = 1.0 | nb12 LOSO | Pooled T hurts women +0.00154 |
| Women alpha = 0.10 (fixed) | W2 | Walk-forward Δ = −0.00017 (not significant) |
| Women HCA = 3.0 | A2 | Δ = −0.00159, 11/14 seasons improved |
| Men HCA = 0.0 | A1 | All HCA values [2, 3, 3.5, 4, 5] hurt men |
| SOS-adjusted MOV | Item 32 | Raw d_MOV better; SOS-adj hurts +0.00065 |
| FTR = FTM/FGA (not FTA/FGA) | Item 4 | FTA/FGA loses by +0.00243 |
| No recency weighting | B1/B2 | All lambda values hurt (uniform best) |
| Dynamic T over pooled T | nb14 | r(eff_sigma, upset_rate) = −0.578, p = 0.005 |
| Play-in games kept in training | Item 16 | Removing hurts +0.00093 |

---

## 14. Reproducibility Checklist

To reproduce predictions for season `S`:

1. **Load data** — `{M|W}RegularSeasonDetailedResults.csv` and the other files listed in §2.
2. **Compute possession estimates** — apply Dean Oliver formula to get `Poss`, `W_OffEff`, `W_DefEff`, etc. (§3.1).
3. **Women only: apply HCA** — adjust raw scores by ±1.5 pts based on `WLoc`, then recompute efficiency columns before SOS iteration (§3.2).
4. **Run SOS iteration** — up to 100 iterations until mean |ΔAdjO| < 1e-4, per-season (§3.3).
5. **Compute all secondary features** — `TeamMOV`, `AstRate`, `CloseWR1/3`, `ConfTourneyWR`, `FTR`, `TOVr` from raw box scores.
6. **Build Elo ratings** — process all games with `Season < S OR (Season == S AND DayNum < 134)` chronologically through the update formula in §5. Use league-specific hyperparameters from §5.3.
7. **Build training matchups** — join tournament results with features, compute differentials, create mirrored rows. Filter to `Season < S`.
8. **Men: train HGB** — fit on `TeamA_margin` target with hyperparameters from §11.
9. **Men: estimate T** — walk-forward procedure over last 8 training seasons (§6.4).
10. **Men: estimate alpha** — walk-forward CV over last 8 seasons using 4-feature HGB and Elo (§6.5).
11. **Women: train logistic regression** — fit on `y` (win/loss) target with hyperparameters from §7.2.
12. **Women: set T = 1.0, alpha = 0.10** — no further calibration needed.
13. **Generate predictions** — for every matchup in the submission template, compute features and apply the formulas in §8.
14. **Verify format** — ID format is `{Season}_{TeamID1}_{TeamID2}` with `TeamID1 < TeamID2`. Pred = P(TeamID1 wins). Clip to [1e-10, 1−1e-10].
