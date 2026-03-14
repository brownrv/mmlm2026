# Experiment Log

Use this file as a lightweight chronological index of notable experiments.

---
## YYYY-MM-DD — <experiment title>

Status: Proposed | Running | Completed | Revisit | Retired

Hypothesis:
- <one sentence>

Dependencies:
- rating_model:<version>
- sim_engine:<version>

MLflow:
- Run name: <name>
- Run ID: <id>

Result:
- <one or two bullets>

Re-test if:
- <trigger condition>

Related:
- docs/experiments/<file>.md
- docs/decisions/<file>.md

---
## 2026-03-14 — LATE-VAL-07 v1: ESPN feature stability audit

Status: Completed

Hypothesis:
- ESPN-derived team features are stable enough by league and season to support deadline-safe late challengers without hidden coverage gaps.

Dependencies:
- feature:late_feat_18_espn_four_factor_v1
- feature:late_feat_19_espn_rotation_v1
- validate:late_val_07_v1

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added a season-by-season ESPN audit exporter and artifacts under `data/interim/espn_audit/`.
- Men ESPN four-factor and rotation features are operationally stable after 2004: overall coverage is `0.944`, median season coverage is `0.988`, and every populated feature column is fully non-null on matched rows.
- Women ESPN four-factor is usable historically from roughly `2017–2025`, but early seasons are weak (`2010–2013` zero coverage, `2014–2016` partial), and current `2026` coverage is still `0.0` in the present raw snapshot.
- Conclusion: ESPN-based men challengers are deadline-safe; women ESPN features are acceptable for historical challenger work but remain live-run blocked until the `2026` ESPN ingest is refreshed.

Re-test if:
- The ESPN parsing or team-spelling join policy changes.
- A new `2026` ESPN boxscore refresh lands and women coverage should be re-audited for live deployment.

Related:
- [scripts/export_espn_audit.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/export_espn_audit.py)
- [espn_audit.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/analysis/espn_audit.py)

---
## 2026-03-14 — MH7-FEAT-01 v1: GLM team quality coefficients

Status: Completed

Hypothesis:
- Per-season OLS team-quality coefficients from regular-season point margins may add orthogonal strength signal beyond the frozen Elo and ESPN-driven leader families.

Dependencies:
- feature:mh7_feat_01_glm_quality_v1
- model:men_reference_margin_generalization_v1
- model:women_late_feat_29_v1

MLflow:
- Run name: `mh7-feat-01-men-glm-quality`
- Run name: `mh7-feat-01-women-glm-quality`

Result:
- Men scored `flat_brier = 0.198515` and `log_loss = 0.585169`, clearly worse than the frozen men leader `0.195566`.
- Women scored `flat_brier = 0.138405` and `log_loss = 0.416720`, far worse than the frozen women leader `0.130381`.
- This branch does not advance in either league; the simple season-level GLM quality coefficients do not combine cleanly with the current frozen leader paths.

Re-test if:
- A future branch uses a richer modeh7-style feature stack or a different downstream model family where GLM quality might interact more constructively.

Related:
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)
- [src/mmlm2026/features/phase_b.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/phase_b.py)

---
## 2026-03-14 — LATE-FEAT-30 v1: men Massey PCA and disagreement

Status: Completed

Hypothesis:
- Season-level Massey consensus structure, expressed as a principal component plus cross-system disagreement, may add residual men team-quality signal beyond the frozen reference margin feature set.

Dependencies:
- feature:late_feat_30_massey_pca_v1
- model:men_reference_margin_generalization_v1

MLflow:
- Run name: `late-feat-30-men-massey-pca`

Result:
- The men challenger scored `flat_brier = 0.198212` and `log_loss = 0.580902` on the 2023–2024 held-out window.
- Per-season flat Brier was `0.207135` in 2023 and `0.189290` in 2024.
- This is clearly worse than the frozen men leader `0.195566`, so `LATE-FEAT-30` does not advance.

Re-test if:
- A future men path uses a materially different rating backbone or a richer Massey-derived representation than simple PCA/disagreement.

Related:
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [src/mmlm2026/features/phase_b.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/phase_b.py)

---
## 2026-03-13 — LATE-FEAT-24 + LATE-FEAT-29 v1: women late-5 plus conference rank

Status: Completed

Hypothesis:
- Combining recent-form signal with conference percentile rank may improve on the standalone women `LATE-FEAT-29` challenger by adding a complementary late-season component.

Dependencies:
- feature:late_feat_24_late5_split_v1
- feature:late_feat_29_conf_pct_rank_v1
- model:women_late_rate_02_v1

MLflow:
- Run name: `late-feat-24-29-women`

Result:
- The combined women challenger scored `flat_brier = 0.131002` and `log_loss = 0.400623` on the 2023–2024 held-out window.
- That is worse than `LATE-FEAT-29` alone (`0.130381`) and also worse than the current women leader candidate margin over the frozen baseline.
- This indicates `late5` does not add useful incremental signal once conference percentile rank is already present on the women ESPN four-factor path.

Re-test if:
- A later women branch changes the base path materially enough that recent-form interactions need to be revisited.

Related:
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)

---
## 2026-03-13 — LATE-FEAT-29 v1: conference percentile rank only

Status: Completed

Hypothesis:
- Within-conference percentile rank may capture residual team-quality signal not fully absorbed by seed, Elo, or ESPN-based strength features.

Dependencies:
- feature:late_feat_29_conf_pct_rank_v1
- model:men_reference_margin_generalization_v1
- model:women_late_rate_02_v1

MLflow:
- Run name: `late-feat-29-men-conf-rank`
- Run name: `late-feat-29-women-conf-rank`

Result:
- Men scored `flat_brier = 0.197564` and `log_loss = 0.580535`, which is still worse than the frozen men leader `0.195566`.
- Women scored `flat_brier = 0.130381` and `log_loss = 0.399624`, narrowly beating the current frozen women leader `0.130427` by `0.000046`.
- The 2025 sanity check also passed: `val-01-2025-holdout-women-late-feat-29` scored `flat_brier = 0.102179`, `log_loss = 0.324663`, `r1_brier = 0.077814`, and `r2plus_brier = 0.127330`.
- That clears `LATE-FEAT-29` as a valid women leader candidate over `late-rate-02-women-espn-four-factor`.

Re-test if:
- A later women branch combines conference percentile rank with another narrow signal and needs a new baseline.

Related:
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)

---
## 2026-03-13 — LATE-FEAT-24 v1: late-5 form split only

Status: Completed

Hypothesis:
- The late-5 offense/defense split may carry useful late-season signal on its own even if the larger late-feature bundle was too noisy.

Dependencies:
- feature:late_feat_24_late5_split_v1
- model:men_reference_margin_generalization_v1
- model:women_late_rate_02_v1

MLflow:
- Run name: `late-feat-24-men-late5-form`
- Run name: `late-feat-24-women-late5-form`

Result:
- Men scored `flat_brier = 0.198986` and `log_loss = 0.584564`, which is still materially worse than the frozen men leader `0.195566`.
- Women scored `flat_brier = 0.131028` and `log_loss = 0.400724`, which is much better than the full late-feature bundle and only `0.000601` behind the frozen women leader `0.130427`.
- This is still a loser in both leagues, but it suggests `LATE-FEAT-24` is the only feature from the 24/27/28/29/31 branch that retained plausible standalone value, especially for women.

Re-test if:
- A follow-up branch pairs late-5 form with only one other narrow feature instead of the full bundle.
- The women ESPN four-factor path gets another close near-miss that could plausibly benefit from a recent-form signal.

Related:
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)
- [src/mmlm2026/features/primary.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/primary.py)

---
## 2026-03-13 — LATE-FEAT bundle v1: late-5, site profile, win-quality, conference rank, and pedigree

Status: Completed

Hypothesis:
- A bundled set of low-cost context features capturing recent form, site performance, win-quality bins, conference standing, and tournament pedigree may improve the current leader families without a broader architecture change.

Dependencies:
- feature:late_feat_24_late5_split_v1
- feature:late_feat_27_site_profiles_v1
- feature:late_feat_28_win_quality_bins_v1
- feature:late_feat_29_conference_percentile_v1
- feature:late_feat_31_program_pedigree_v1
- model:men_reference_margin_generalization_v1
- model:women_late_rate_02_v1

MLflow:
- Run name: `late-feat-bundle-24-27-28-29-31-men`
- Run name: `late-feat-bundle-24-27-28-29-31-women`

Result:
- Added eight matchup differentials: `late5_off_diff`, `late5_def_diff`, `road_win_pct_diff`, `neutral_net_eff_diff`, `close_win_pct_5_diff`, `blowout_win_pct_diff`, `conf_pct_rank_diff`, and `pedigree_score_diff`.
- Men scored `flat_brier = 0.200470` and `log_loss = 0.590627`, materially worse than the frozen men leader `0.195566`.
- Women scored `flat_brier = 0.133791` and `log_loss = 0.406771`, also worse than the frozen women leader `0.130427`.
- This bundled late-feature slice does not advance in either league.

Re-test if:
- A later branch isolates one of these features and shows a clear single-feature gain.
- Better external data supports a narrower version of the site or pedigree features.

Related:
- [src/mmlm2026/features/primary.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/primary.py)
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)
- [tests/test_primary_features.py](/c:/Users/brown/Documents/GitHub/mmlm2026/tests/test_primary_features.py)
- [tests/test_women_round_group_routing.py](/c:/Users/brown/Documents/GitHub/mmlm2026/tests/test_women_round_group_routing.py)

---
## 2026-03-13 — LATE-ARCH-META-01 v1: logit-Ridge meta-learner

Status: Completed

Hypothesis:
- A Ridge meta-learner on logit-transformed OOF base predictions may beat the frozen single-model leaders by combining near-miss challenger members more effectively than the current direct model-selection path.

Dependencies:
- architecture:late_arch_meta_01_v1
- model:men_reference_margin_generalization_v1
- model:women_late_rate_02_v1
- challenger:late_rate_01_v2
- challenger:late_feat_26
- challenger:late_feat_23

MLflow:
- Run name: `late-arch-meta-01-men`
- Run name: `late-arch-meta-01-women`

Result:
- Built a reusable logit-Ridge meta stack over regenerated out-of-season base predictions.
- Men members: frozen reference margin model, ESPN-strength near miss, and Pythagorean near miss.
- Women members: frozen ESPN four-factor leader, routed-basic predecessor, and seed-gap near miss.
- Women stack scored `flat_brier = 0.135684` and `log_loss = 0.413501`, which is clearly worse than the frozen women leader `0.130427`.
- Men stack scored `flat_brier = 0.205593` and `log_loss = 0.676478`, dramatically worse than the frozen men leader `0.195566`.
- The current near-miss base members are too correlated or too weak for this Ridge meta-learner to help.

Re-test if:
- A later queue item produces materially more diverse base members.
- The ensemble member set changes enough to justify another stacked meta test.

Related:
- [src/mmlm2026/evaluation/meta.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/evaluation/meta.py)
- [scripts/run_logit_ridge_meta.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_logit_ridge_meta.py)
- [tests/test_meta.py](/c:/Users/brown/Documents/GitHub/mmlm2026/tests/test_meta.py)

---
## 2026-03-13 — LATE-ARCH-DW-01 v1: decay-weighted training

Status: Completed

Hypothesis:
- Upweighting recent tournament seasons during training may help the current frozen leader families adapt to modern tournament dynamics without adding new data.

Dependencies:
- model:men_reference_margin_generalization_v1
- model:women_late_rate_02_v1
- architecture:late_arch_dw_01_v1

MLflow:
- Run name: `late-arch-dw-01-men`
- Run name: `late-arch-dw-01-women`

Result:
- Applied exponential season-recency decay (`decay_base = 0.9`) to the current men and women leader families.
- Men scored `flat_brier = 0.199205` and `log_loss = 0.584961`, materially worse than the frozen men leader `0.195566`.
- Women scored `flat_brier = 0.132470` and `log_loss = 0.407918`, which was closer but still behind the frozen women leader `0.130427`.
- This simple recency-weighted training scheme does not advance in either league.

Re-test if:
- A later branch combines decay weighting with a different architecture or external data source.
- The validation window changes materially enough that modern-season weighting should be revisited.

Related:
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)

---
## 2026-03-13 — LATE-EXT-03 v1: ESPN component-level situational decomposition

Status: Completed

Hypothesis:
- Exposing ESPN four-factor components individually, instead of only the composite strength score, may let the current men and women leader families use richer offensive and defensive situational structure.

Dependencies:
- dataset:espn_processed_v1
- feature:late_feat_18_espn_four_factor_v1
- model:men_reference_margin_generalization_v1
- model:women_late_rate_02_v1

MLflow:
- Run name: `late-ext-03-men-espn-components`
- Run name: `late-ext-03-women-espn-components`

Result:
- Added component-level ESPN differentials for `efg`, `tov_rate`, `orb_pct`, `ftr`, and the corresponding opponent-side defensive components on top of the current frozen-path challengers.
- Men scored `flat_brier = 0.199281` and `log_loss = 0.578924`, which is materially worse than the frozen men leader `0.195566`.
- Women scored `flat_brier = 0.136654` and `log_loss = 0.416621`, which is also clearly worse than the frozen women leader `0.130427`.
- This richer ESPN decomposition does not combine cleanly with the current frozen leader families and should not advance.

Re-test if:
- ESPN parsing gains materially better historical coverage or cleaner player/game joins.
- A later architecture is explicitly designed to absorb correlated component-level features.

Related:
- [src/mmlm2026/features/primary.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/primary.py)
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)

---
## 2026-03-13 — LATE-RATE-02 v1: women ESPN four-factor latent strength

Status: Completed

Hypothesis:
- A stronger women team-strength input from ESPN-derived four-factor summaries can improve the routed women leader without reopening the broader women feature stack.

Dependencies:
- frozen_models:women_routed_round_group_v1
- dataset:espn_processed_women_v1
- validation:late_val_07_women_routed_holdout

MLflow:
- Run name: `late-rate-02-women-espn-four-factor`
- Run name: `val-01-2025-holdout-women-late-rate-02`

Result:
- Added `espn_four_factor_strength_diff` to the routed women `R1` vs `R2+` challenger using leakage-safe ESPN women boxscore matches against Kaggle regular-season games.
- The 2023-2024 held-out run scored `flat_brier = 0.130427` and `log_loss = 0.400684`, beating the current routed women leader `0.131950`.
- The 2025 sanity check also passed with `flat_brier = 0.103438`, `log_loss = 0.328220`, `r1_brier = 0.081535`, and `r2plus_brier = 0.126049`, which is better than the prior routed women sanity check `0.106055`.
- This clears `LATE-RATE-02 v1` to replace `LATE-ARCH-RG-08` as the active women leader.

Re-test if:
- ESPN women coverage, team mapping, or matching rules change.
- A later women challenger beats the same held-out and 2025-sanity protocol.

Related:
- [src/mmlm2026/features/espn.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/espn.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)
- [tests/test_espn_features.py](/c:/Users/brown/Documents/GitHub/mmlm2026/tests/test_espn_features.py)

---
## 2026-03-13 — LATE-MKT-01: men BetExplorer market-strength challenger

Status: Completed

Hypothesis:
- A regular-season market-implied strength feature derived from BetExplorer pregame probabilities will improve the frozen men reference path by adding compressed public-strength information not fully captured by Elo and seed.

Dependencies:
- frozen_models:men_generalization_reference_margin
- dataset:betexplorer_processed_v1
- validation:late_val_06_market_audit

MLflow:
- Run name: `late-mkt-01-men-market-strength`
- Run ID: N/A

Result:
- Added `market_implied_strength_diff`, built from season-level average logit-transformed BetExplorer pregame win probabilities on regular-season games before DayNum `134`.
- The men challenger scored `flat_brier = 0.197276` and `log_loss = 0.579413` on the 2023–2024 held-out window.
- Per-season flat Brier: `2023 = 0.207453`, `2024 = 0.187099`.
- This is worse than the frozen men leader `0.195566`, so the first men market-strength slice does not advance.

Re-test if:
- A second market formulation uses a different aggregation, such as closing/opening split, market residuals vs Elo, or opponent-adjusted market strength.
- The BetExplorer dataset is enriched with additional fields beyond the current single implied-probability snapshot.

Related:
- [src/mmlm2026/features/primary.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/primary.py)
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [data/interim/reference_runs/m_reference_margin/validation/per_season_metrics.csv](/c:/Users/brown/Documents/GitHub/mmlm2026/data/interim/reference_runs/m_reference_margin/validation/per_season_metrics.csv)

---
## 2026-03-13 — LATE-VAL-06: BetExplorer market-data coverage and leakage audit

Status: Completed

Hypothesis:
- BetExplorer odds can serve as a usable late challenger only if the historical joins are clean, the implied probabilities behave like genuine pregame probabilities, and coverage is high enough in the seasons and leagues we care about.

Dependencies:
- dataset:betexplorer_processed_v1
- diagnostics:frozen_historical_performance_v1
- plan:002_late_val_06

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added a market-audit workflow that loads canonical BetExplorer parquet files, summarizes per-season coverage/sanity, and compares market-only tournament Brier to the frozen leaders on the exact covered rows.
- Men coverage is operationally usable in recent seasons: regular-season coverage is `98.96%` (`2021`), `99.36%` (`2022`), `99.75%` (`2023`), `99.22%` (`2024`), and `97.45%` (`2025`); tournament coverage is `100%` for `2021` through `2025`.
- Women coverage is not operationally viable: regular-season coverage is `0%` for `1998-2024`, then only `25.28%` in `2025`; tournament coverage is `0%` for `1998-2024`, then `85.07%` in `2025` only.
- On the market-covered tournament subset, men market-only Brier is slightly worse than the frozen local men leader overall (`0.186486` vs `0.184178` across `940` games), though it beats the local model in a few seasons (`2013`, `2016`, `2021`, `2022`, `2023`).
- On women, the market-covered subset is only `2025`, but there it beats the frozen local women leader materially (`0.102316` vs `0.126317` across `53` games). That result is interesting but not enough to overcome the historical coverage gap.
- Verdict: `LATE-MKT-01` is green-lit for men only. Women market challengers should remain blocked unless BetExplorer historical coverage expands or a different market source is added.

Re-test if:
- BetExplorer women coverage is backfilled for pre-2025 seasons.
- A new market dataset with better women history is added.
- The frozen leaders change enough that the market-covered subset comparison should be re-run.

Related:
- [src/mmlm2026/analysis/market_audit.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/analysis/market_audit.py)
- [scripts/export_market_audit.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/export_market_audit.py)
- [data/interim/market_audit/betexplorer_viability_summary.csv](/c:/Users/brown/Documents/GitHub/mmlm2026/data/interim/market_audit/betexplorer_viability_summary.csv)
- [data/interim/market_audit/betexplorer_tourney_vs_local.csv](/c:/Users/brown/Documents/GitHub/mmlm2026/data/interim/market_audit/betexplorer_tourney_vs_local.csv)

---
## 2026-03-13 — LATE-EXT-01/02: benchmark-guided challenger design

Status: Completed

Hypothesis:
- Comparing the frozen leaders against the `adj_quality_gap_v10` all-pairs benchmark on the same matchup universe will identify recurring round-bucket cells where local challengers are most worth trying next.

Dependencies:
- frozen_models:men_generalization_reference_margin
- frozen_models:women_routed_round_group_v1
- diagnostics:all_matchups_export_v1
- dataset:data_reference_v1

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added benchmark-gap export tooling to compare local all-matchups predictions against `data/reference` and summarize recurring gaps by league, round group, and matchup-likelihood bucket.
- The highest-priority recurring gap is women `R2+ / very_likely`: `played_games = 323`, `play_prob_mass = 321.6`, local `flat_brier = 0.166778`, benchmark `flat_brier = 0.145030`, `mean_brier_gap = 0.021748`.
- The second strongest recurring gap is women `R1 / definite`: `played_games = 432`, `play_prob_mass = 432.0`, local `flat_brier = 0.117087`, benchmark `flat_brier = 0.109559`, `mean_brier_gap = 0.007529`.
- Men gaps are smaller and later-round concentrated: `R2+ / likely` (`mean_brier_gap = 0.003798`) and `R2+ / plausible` (`0.009895`) are the only recurring cells with meaningful mass.
- Recommendation: keep `data/reference` diagnostic-only, and use these cells to design the next targeted challengers rather than treating benchmark probabilities as trainable features or replacement selection metrics.

Re-test if:
- The frozen leaders change.
- `data/reference` is refreshed with a different benchmark model or season coverage.
- A new challenger meaningfully changes women `R2+ / very_likely` or men `R2+ likely/plausible` behavior.

Related:
- [src/mmlm2026/analysis/benchmark_gap.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/analysis/benchmark_gap.py)
- [scripts/export_benchmark_gap_analysis.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/export_benchmark_gap_analysis.py)
- [data/interim/benchmark_gap_analysis/benchmark_gap_priority_cells.csv](/c:/Users/brown/Documents/GitHub/mmlm2026/data/interim/benchmark_gap_analysis/benchmark_gap_priority_cells.csv)

---
## 2026-03-13 — LATE-FEAT-25: season momentum

Status: Completed

Hypothesis:
- Teams improving across the full season arc should outperform flat or declining teams at tournament time, so second-half minus first-half average margin adds signal beyond seed and Elo.

Dependencies:
- frozen_models:men_generalization_reference_margin
- frozen_models:women_routed_round_group_v1
- feature:late_feat_25_season_momentum_v1

MLflow:
- Run name: `late-feat-25-men-season-momentum`
- Run name: `late-feat-25-women-season-momentum`

Result:
- Added `season_momentum_diff`, defined as second-half minus first-half average margin with a midseason split at DayNum `67`, to the current men and women leader-family challengers.
- The men challenger scored `flat_brier = 0.196529` and `log_loss = 0.578522` on the 2023-2024 held-out window, which is still worse than the frozen men leader `0.195566`.
- The women challenger scored `flat_brier = 0.132935` and `log_loss = 0.407150`, which improved on some recent women late-feature slices but still stayed behind the routed women leader `0.131950`.
- Season momentum does not advance as a standalone additive feature in either league.

Re-test if:
- A future challenger uses season momentum only in routed `R2+` models.
- The split definition changes from a fixed midseason day to a team-specific half-season split.

Related:
- [src/mmlm2026/features/primary.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/primary.py)
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)

---
## 2026-03-13 — LATE-FEAT bundle: Elo momentum + Pythagorean expectancy + seed-Elo gap

Status: Completed

Hypothesis:
- The three low-effort derived features may be too weak individually but useful together, because they summarize different views of team quality drift, score-based strength, and seed-vs-rating mispricing.

Dependencies:
- frozen_models:men_generalization_reference_margin
- frozen_models:women_routed_round_group_v1
- feature:late_feat_22_elo_momentum_v1
- feature:late_feat_26_pythag_expectancy_v1
- feature:late_feat_23_seed_elo_gap_v1

MLflow:
- Run name: `late-feat-bundle-men-elo-pythag-seed-gap`
- Run name: `late-feat-bundle-women-elo-pythag-seed-gap`

Result:
- Tested the planned bundled challenger using `elo_momentum_diff`, `pythag_diff`, and `seed_elo_gap_diff` together on top of the current frozen leader families.
- The men bundle scored `flat_brier = 0.196494` and `log_loss = 0.577848`, which is worse than the best individual low-effort men slices and still behind the frozen men leader `0.195566`.
- The women bundle scored `flat_brier = 0.138380` and `log_loss = 0.418886`, which is materially worse than the routed women leader `0.131950`.
- The combined low-effort bundle does not advance and suggests these derived features are not interacting constructively in the current leader families.

Re-test if:
- A future challenger uses these features only in a routed `R2+` model rather than a unified path.
- The bundle is paired with a different upstream rating family instead of the current frozen leaders.

Related:
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)

---
## 2026-03-13 — LATE-FEAT-23: seed-Elo gap

Status: Completed

Hypothesis:
- Teams whose Elo materially exceeds or trails what their seed line implies are mispriced by seed-based tournament models, so `seed_elo_gap_diff` should add signal on top of seed and Elo separately.

Dependencies:
- frozen_models:men_generalization_reference_margin
- frozen_models:women_routed_round_group_v1
- feature:late_feat_23_seed_elo_gap_v1

MLflow:
- Run name: `late-feat-23-men-seed-elo-gap`
- Run name: `late-feat-23-women-seed-elo-gap`

Result:
- Added `seed_elo_gap_diff`, defined as actual Elo minus the seed-implied Elo baseline `1750 - (seed - 1) * 25`, to the current men and women leader-family challengers.
- The men challenger scored `flat_brier = 0.197100` and `log_loss = 0.580906` on the 2023-2024 held-out window, which was materially worse than the frozen men leader at `0.195566`.
- The women challenger scored `flat_brier = 0.132013` and `log_loss = 0.405026`, improving over several recent women late-feature challengers but still missing the routed women leader `0.131950` by `0.000063`.
- Seed-Elo gap does not advance as a standalone additive feature in either league.

Re-test if:
- It is bundled with Elo momentum and/or Pythagorean expectancy instead of tested in isolation.
- A future routed challenger uses the seed-Elo gap only in `R2+` where seeding mispricing may matter more.

Related:
- [src/mmlm2026/features/elo.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/elo.py)
- [src/mmlm2026/features/primary.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/primary.py)
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)

---
## 2026-03-13 — LATE-FEAT-26: Pythagorean expectancy

Status: Completed

Hypothesis:
- Pythagorean win expectancy from season-average scoring and points allowed is a more stable quality signal than raw win percentage and adds useful information on top of seed and Elo.

Dependencies:
- frozen_models:men_generalization_reference_margin
- frozen_models:women_routed_round_group_v1
- feature:late_feat_26_pythag_expectancy_v1

MLflow:
- Run name: `late-feat-26-men-pythag`
- Run name: `late-feat-26-women-pythag`

Result:
- Added `pythag_diff`, defined as the differential in season-level Pythagorean expectancy using the standard `10.25` exponent, to the current men and women leader-family challengers.
- The men challenger scored `flat_brier = 0.195654` and `log_loss = 0.578176` on the 2023-2024 held-out window, nearly tying but still missing the frozen men leader at `0.195566`.
- The women challenger scored `flat_brier = 0.133332` and `log_loss = 0.409086`, improving over several recent women late-feature challengers but still remaining behind the routed women leader at `0.131950`.
- Pythagorean expectancy does not advance as a standalone additive feature in either league.

Re-test if:
- It is bundled with another low-effort derived feature such as seed-Elo gap or Elo momentum rather than tested in isolation.
- A routed `R2+`-specific challenger uses Pythagorean expectancy only in later rounds.

Related:
- [src/mmlm2026/features/primary.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/primary.py)
- [scripts/run_men_reference_margin.py](/c:/Users\brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users\brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)

---
## 2026-03-13 — LATE-FEAT-22: Elo momentum

Status: Completed

Hypothesis:
- Elo trajectory from `DayNum 115` to the end of the regular season adds signal beyond static end-of-season Elo, especially for teams trending up or down entering the tournament.

Dependencies:
- frozen_models:men_generalization_reference_margin
- frozen_models:women_routed_round_group_v1
- feature:late_feat_22_elo_momentum_v1

MLflow:
- Run name: `late-feat-22-men-elo-momentum`
- Run name: `late-feat-22-women-elo-momentum`

Result:
- Added `elo_momentum_diff`, defined as pre-tournament Elo minus `DayNum 115` Elo, to the current men and women leader-family challengers.
- The men challenger scored `flat_brier = 0.195764` and `log_loss = 0.578253` on the 2023-2024 held-out window, improving on `LATE-FEAT-21` but still missing the frozen men leader at `0.195566`.
- The women challenger scored `flat_brier = 0.138076` and `log_loss = 0.418725`, well behind the routed women leader at `0.131950`.
- Elo momentum does not advance as a unified additive feature for either league.

Re-test if:
- A future challenger uses Elo momentum only in routed `R2+` models instead of unified models.
- Elo momentum is combined with another low-effort derived bundle such as seed-Elo gap or Pythagorean expectancy rather than tested in isolation.

Related:
- [src/mmlm2026/features/elo.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/elo.py)
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)

---
## 2026-03-13 — LATE-FEAT-21: tournament-only Elo

Status: Completed

Hypothesis:
- A second Elo system trained only on historical tournament games captures persistent tournament-specific strength that improves the frozen leaders when added as `tourney_elo_diff`.

Dependencies:
- frozen_models:men_generalization_reference_margin
- frozen_models:women_routed_round_group_v1
- feature:late_feat_21_tourney_elo_v1

MLflow:
- Run name: `late-feat-21-men-tourney-elo`
- Run name: `late-feat-21-women-tourney-elo`

Result:
- Added a tournament-only Elo feature path that fits a separate Elo process on historical tournament games and carries `tourney_elo_diff` as an additive challenger feature.
- The men challenger scored `flat_brier = 0.196078` and `log_loss = 0.578960` on the 2023-2024 held-out window, which did not beat the frozen men leader at `0.195566`.
- The women challenger scored `flat_brier = 0.136614` and `log_loss = 0.418370` on the 2023-2024 held-out window, which did not beat the frozen women leader at `0.131950`.
- Tournament-only Elo did not advance in either league.

Re-test if:
- A future challenger uses tournament-only Elo as a routed `R2+`-specific signal rather than a unified additive feature.
- The tournament-only Elo process is retuned jointly with the base full-season Elo rather than added post hoc.

Related:
- [src/mmlm2026/features/elo.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/elo.py)
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)

---
## 2026-03-13 — LATE-FEAT-19: men ESPN rotation-stability proxies

Status: Completed

Hypothesis:
- ESPN-derived player/rotation continuity proxies layered onto the strongest men latent-strength challenger will improve held-out flat Brier beyond the ESPN four-factor slice alone.

Dependencies:
- frozen_models:men_generalization_reference_margin
- feature:late_rate_01_ridge_strength_v1
- feature:late_feat_18_espn_four_factor_v1
- feature:late_feat_19_espn_rotation_v1

MLflow:
- Run name: `late-feat-19-men-espn-rotation`

Result:
- Added a leakage-safe ESPN rotation-stability feature path using matched regular-season ESPN boxscores only.
- Derived team-season proxies for top-5 minute share, top-player minute share, most-common starter-lineup stability, and a composite `espn_rotation_stability`.
- Layering `espn_rotation_stability_diff` on top of the ridge + ESPN four-factor men challenger degraded the held-out window to `flat_brier = 0.196864` and `log_loss = 0.577258`.
- This is worse than both the frozen men leader (`0.195566`) and the ESPN four-factor slice (`0.195652`), so the rotation proxy does not advance.

Re-test if:
- ESPN player data is extended with cleaner availability signals such as missed-game streaks, top-player absence flags, or lineup continuity over only the last 10 games.
- A future challenger uses these rotation proxies in a routed `R2+`-specific model instead of the unified men reference path.

Related:
- [src/mmlm2026/features/espn.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/espn.py)
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)

---
## 2026-03-13 — LATE-RATE-01 v2: men ESPN four-factor strength

Status: Completed

Hypothesis:
- ESPN-derived four-factor team strength, matched only to Kaggle regular-season games, will improve the frozen men reference margin model when added to the latent-strength stack.

Dependencies:
- frozen_models:men_generalization_reference_margin
- feature:late_rate_01_ridge_strength_v1
- feature:late_feat_18_espn_four_factor_v1
- feature:elo_tuned_carryover_men_v1

MLflow:
- Run name: `late-rate-01-men-espn-four-factor`

Result:
- Added a leakage-safe ESPN men four-factor feature path by mapping ESPN team IDs to Kaggle TeamIDs and retaining only ESPN games that match Kaggle regular-season results.
- The second `LATE-RATE-01` slice added `espn_four_factor_strength_diff` on top of the first ridge-strength variant.
- The challenger scored `flat_brier = 0.195652` and `log_loss = 0.576072` on the 2023-2024 held-out window.
- This is effectively tied with the frozen men leader but still slightly worse than `0.195566`, so it does not advance.

Re-test if:
- ESPN-derived player-continuity or richer four-factor variants are added.
- A decay-weighted or tournament-only upstream rating is combined with the ESPN strength layer.

Related:
- [src/mmlm2026/features/espn.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/features/espn.py)
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)

---
## 2026-03-13 — LATE-RATE-01 v1: men ridge-strength latent rating

Status: Completed

Hypothesis:
- A ridge-regularized season-level team strength rating built from pre-tournament detailed results will improve the frozen men reference margin model more than another calibration-only tweak.

Dependencies:
- frozen_models:men_generalization_reference_margin
- feature:phase_b_v1
- feature:elo_tuned_carryover_men_v1

MLflow:
- Run name: `late-rate-01-men-ridge-strength`

Result:
- Added `ridge_strength` to Phase B team features via a season-level ridge rating fit on regular-season margin-per-100 results with a home-court term.
- Tested the first `LATE-RATE-01` slice by adding `ridge_strength_diff` to the existing men reference margin pipeline.
- The challenger scored `flat_brier = 0.196103` and `log_loss = 0.578436` on the 2023-2024 held-out window.
- The frozen men leader remains better at `0.195566`, so this first latent-strength variant does not advance.

Re-test if:
- A richer upstream men strength layer is built from ESPN-derived four-factor or player-continuity features.
- A second latent-strength variant changes the rating target materially (for example offense/defense or decay-weighted rating construction).

Related:
- [scripts/run_men_reference_margin.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_reference_margin.py)
- [docs/roadmaps/PLAN-002-competition-attack-plan.md](/c:/Users/brown/Documents/GitHub/mmlm2026/docs/roadmaps/PLAN-002-competition-attack-plan.md)

---
## 2026-03-13 — LATE-ARCH-RG-08: routed women `R1` vs `R2+` model

Status: Completed

Hypothesis:
- A true routed women model with separate `R1` and `R2+` training paths will improve held-out flat Brier over the unified `ARCH-04B` baseline.

Dependencies:
- frozen_models:women_arch04b_tuned_elo
- diagnostics:frozen_historical_performance
- arch-rg

MLflow:
- Run name: `late-arch-rg-08-women-routed-round-group`

Result:
- Added `scripts/run_women_routed_round_group_model.py` to train separate women logistic paths for `R1` and `R2+`, with round-group routing at validation and inference time plus a unified fallback for unroutable rows.
- The routed challenger scored `flat_brier = 0.131950` on the 2023-2024 held-out window, narrowly beating the frozen women leader `ARCH-04B` at `0.132193`.
- The 2025 sanity check also passed with `flat_brier = 0.106055` and `log_loss = 0.334307`, so this challenger is cleared to replace `ARCH-04B` as the active women leader.

Re-test if:
- The routed women model fails the 2025 sanity check.
- A later women challenger beats this routed model on the same held-out protocol.

Related:
- [scripts/run_women_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_routed_round_group_model.py)
- [docs/roadmaps/PLAN-002-competition-attack-plan.md](/c:/Users/brown/Documents/GitHub/mmlm2026/docs/roadmaps/PLAN-002-competition-attack-plan.md)

---
## 2026-03-13 — LATE-ARCH-RG-07: routed men `R1` vs `R2+` model

Status: Completed

Hypothesis:
- A true routed men model with separate `R1` and `R2+` training paths will improve held-out flat Brier where round-group-specific calibration alone did not.

Dependencies:
- frozen_models:men_generalization_reference_margin
- diagnostics:frozen_historical_performance
- arch-rg

MLflow:
- Run name: `late-arch-rg-07-men-routed-round-group`

Result:
- Added `scripts/run_men_routed_round_group_model.py` to train separate men margin-regression paths for `R1` and `R2+`, with round-group routing at validation and inference time plus a unified fallback for unroutable rows.
- The routed challenger scored `flat_brier = 0.197102` on the 2023-2024 held-out window, which still did not beat the frozen men leader at `0.195566`.
- Conclusion: the men round-group split is real enough to justify analysis, but a full routed model still did not improve on the frozen men reference path. Do not promote it.

Re-test if:
- The men latent strength layer changes materially before submission.
- A later routed men challenger uses external data or a different upstream rating family.

Related:
- [scripts/run_men_routed_round_group_model.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_routed_round_group_model.py)
- [docs/roadmaps/PLAN-002-competition-attack-plan.md](/c:/Users/brown/Documents/GitHub/mmlm2026/docs/roadmaps/PLAN-002-competition-attack-plan.md)

---
## 2026-03-13 — ARCH-04C: ARCH-04B + women_hca_adj_qg_diff challenger

Status: Completed

Hypothesis:
- The frozen women leader `ARCH-04B` may improve if it absorbs the strongest women-specific structural strength feature, `women_hca_adj_qg_diff`, without changing model family or adding broader complexity.

Dependencies:
- frozen_models:women_arch04b_tuned_elo
- feature:feat_13_women_hca_adj_qg_v1

MLflow:
- Run name: `arch-04c-arch04b-plus-adj-qg-women`

Result:
- Added `scripts/run_arch04b_adj_qg_challenger.py` to fit a direct `ARCH-04B` extension using `seed_diff`, tuned `elo_diff`, and `women_hca_adj_qg_diff`.
- The challenger scored `flat_brier = 0.140032` on the 2023-2024 held-out window, which is materially worse than the frozen women leader `ARCH-04B` at `0.132193`.
- Conclusion: the women HCA-adjusted quality-gap signal does not combine cleanly with the frozen tuned-Elo baseline in this narrow structural form. It should not advance.

Re-test if:
- The women HCA-adjusted quality-gap feature definition changes materially.
- The frozen women baseline is replaced before submission.

Related:
- [scripts/run_arch04b_adj_qg_challenger.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_arch04b_adj_qg_challenger.py)
- [docs/roadmaps/PLAN-002-competition-attack-plan.md](/c:/Users/brown/Documents/GitHub/mmlm2026/docs/roadmaps/PLAN-002-competition-attack-plan.md)

---
## 2026-03-13 — VAL-07: women very_likely / R2+ calibration challenger

Status: Completed

Hypothesis:
- The frozen women model's weakness in `very_likely` and later-round games may be reducible with a narrow temperature-scaling layer targeted at `very_likely` and `R2+` rows without changing the base women model.

Dependencies:
- frozen_models:women_arch04b_tuned_elo
- diagnostics:frozen_historical_performance

MLflow:
- Run name: `val-07-women-bucket-group-calibration`

Result:
- Added `scripts/run_women_bucket_group_calibration.py` to fit a global women temperature plus targeted overrides for `very_likely` and `R2+` groups derived from bracket play-probability buckets.
- The challenger scored `flat_brier = 0.134298` on the 2023-2024 held-out window, which did not beat the frozen women leader `ARCH-04B` at `0.132193`.
- The fitted `very_likely` temperature reverted to the global value (`~1.1588`), while `R2+` moved slightly higher (`~1.2573`), so the historical weakness appears to be more round-driven than bucket-driven under the current women model.

Re-test if:
- The frozen women model is replaced before submission.
- Women bucket diagnostics change materially after the 2026 data refresh.

Related:
- [scripts/run_women_bucket_group_calibration.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_women_bucket_group_calibration.py)
- [notebooks/frozen_historical_performance_tables.ipynb](/c:/Users/brown/Documents/GitHub/mmlm2026/notebooks/frozen_historical_performance_tables.ipynb)

---
## 2026-03-13 — VAL-06: men round-group calibration challenger

Status: Completed

Hypothesis:
- The frozen men model's persistent `R2+` weakness may be fixable with round-group-specific calibration and Elo blend weights without changing the underlying feature stack.

Dependencies:
- frozen_models:men_generalization_reference_margin
- diagnostics:frozen_historical_performance

MLflow:
- Run name: `val-06-men-round-group-calibration`

Result:
- Added `scripts/run_men_round_group_calibration.py` to fit separate prior-season calibration states for `R1` and `R2+` on top of the frozen men model.
- The challenger scored `flat_brier = 0.197124` on the 2023-2024 held-out window, which did not beat the frozen men leader at `0.195566`.
- The fitted states diverged materially (`R1 alpha ~ 0.72`, `R2+ alpha ~ 0.35`), which confirms the diagnostic pattern is real, but the net gain was not enough to replace the frozen model.

Re-test if:
- A later men challenger changes the base men probability surface before submission.
- Historical round-group diagnostics move materially after the 2026 data refresh.

Related:
- [scripts/run_men_round_group_calibration.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/run_men_round_group_calibration.py)
- [notebooks/frozen_historical_performance_tables.ipynb](/c:/Users/brown/Documents/GitHub/mmlm2026/notebooks/frozen_historical_performance_tables.ipynb)

---
## 2026-03-13 — Gate 3: historical frozen-model diagnostics export and notebook

Status: Completed

Hypothesis:
- A saved multi-season realized-game table for the frozen models will make it easier to inspect systematic weak spots by league, round, bucket, and season without recomputing models inside notebooks.

Dependencies:
- plan:002_gate3
- frozen_models:men_generalization_reference_margin
- frozen_models:women_arch04b_tuned_elo

Result:
- Added a reusable frozen-model scoring module plus `scripts/export_frozen_historical_performance.py` to materialize historical played-game diagnostics to `data/interim/frozen_historical_performance.parquet` and `.csv`.
- Added `notebooks/frozen_historical_performance_tables.ipynb` with season/league/round/bucket tables, heatmaps, bucket calibration views, total Brier-contribution bars, and worst-cell diagnostics for improvement hunting.

Re-test if:
- The frozen men or women model is replaced before final submission.
- Bucket policy or round assignment changes.

Related:
- [scripts/export_frozen_historical_performance.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/export_frozen_historical_performance.py)
- [notebooks/frozen_historical_performance_tables.ipynb](/c:/Users/brown/Documents/GitHub/mmlm2026/notebooks/frozen_historical_performance_tables.ipynb)
- [docs/roadmaps/PLAN-002-competition-attack-plan.md](/c:/Users/brown/Documents/GitHub/mmlm2026/docs/roadmaps/PLAN-002-competition-attack-plan.md)

---
## 2026-03-13 — VAL-03: frozen-pair bracket diagnostics on historical 2025 dry run

Status: Completed

Hypothesis:
- Deterministic bracket diagnostics on the frozen men and women models should not reveal an obvious calibration failure in the high-likelihood matchup buckets.

Dependencies:
- plan:002_gate3
- arch-04b
- val-05
- bracket_dp:v1

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Ran the combined frozen submission builder on the historical 2025 combined sample and wrote a validated output at [2025_val-03-2025_submission.csv](/c:/Users/brown/Documents/GitHub/mmlm2026/data/submissions/2025_val-03-2025_submission.csv), with bracket artifacts under [2025_val-03-2025_artifacts](/c:/Users/brown/Documents/GitHub/mmlm2026/data/submissions/2025_val-03-2025_artifacts).
- Men high-likelihood buckets were acceptable: `definite` games scored `flat_brier = 0.131962` over 32 games and `very_likely` games scored `0.163090` over 25 games. The tiny `plausible` and `remote` buckets were noisier but low-volume.
- Women high-likelihood buckets were also stable: `definite` games scored `flat_brier = 0.112011` over 32 games and `very_likely` games scored `0.112525` over 30 games.
- Conclusion: the frozen pair does not show an obvious bracket-space calibration risk in the high-likelihood buckets. `VAL-03` is operationally satisfied on the historical dry-run path.

Re-test if:
- The frozen men or women model changes.
- The bracket DP bucket definitions change.
- The real 2026 Stage 2 run produces materially different high-likelihood bucket behavior.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — Gate 3: historical frozen-submission dry run

Status: Completed

Hypothesis:
- The frozen men and women models can be assembled into one validated Kaggle-format submission without changing any modeling assumptions at Stage 2 time.

Dependencies:
- plan:002_gate3
- arch-04b
- val-05
- submission_validation:v1

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added a combined frozen-submission builder at [scripts/build_frozen_submission.py](/c:/Users/brown/Documents/GitHub/mmlm2026/scripts/build_frozen_submission.py) plus reusable submission-ID helpers in [src/mmlm2026/submission/frozen.py](/c:/Users/brown/Documents/GitHub/mmlm2026/src/mmlm2026/submission/frozen.py).
- A historical 2025 combined sample with `4556` rows was generated and the new builder produced a fully validated submission at [2025_gate3-dryrun-2025_submission.csv](/c:/Users/brown/Documents/GitHub/mmlm2026/data/submissions/2025_gate3-dryrun-2025_submission.csv), with bracket artifacts saved under [2025_gate3-dryrun-2025_artifacts](/c:/Users/brown/Documents/GitHub/mmlm2026/data/submissions/2025_gate3-dryrun-2025_artifacts).
- The live 2026 Stage 2 sample cannot yet be scored end to end from the current raw snapshot because 2026 regular-season inputs are not present locally. The remaining Gate 3 work is therefore operational: rerun the frozen builder once the 2026 data snapshot lands.

Re-test if:
- 2026 regular-season and seed data are refreshed locally.
- The frozen men or women leader changes.
- Kaggle changes the Stage 2 sample-submission schema or TeamID universe.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — VAL-05: replication-vs-generalization Elo parameter comparison

Status: Completed

Hypothesis:
- Generalization-tuned Elo parameters may produce better held-out downstream model performance than replication-tuned parameters, especially on the men reference path.

Dependencies:
- plan:002_gate2
- val-05
- feature:elo_tuned_carryover_men_local_v1
- feature:elo_tuned_carryover_women_local_v1
- validation:leave_season_out_v1

MLflow:
- Run name: val-05-men-reference-margin-generalization
- Run ID: 337d3b992b884dbb800c078561d37622
- Run name: val-05-women-reference-replication-generalization
- Run ID: fdf6e7b90d07447ab6624b51d3be06ca

Result:
- Men generalization tuning selected a much lower-volatility Elo regime than replication tuning and improved the downstream held-out men reference model from `flat_brier = 0.195985` to `0.195566`.
- Women generalization tuning selected a materially different Elo regime, but the downstream held-out women reference model worsened slightly from `flat_brier = 0.138383` to `0.138507`; the earlier replication-style women run at `0.138132` remains the best of that family.
- Conclusion: the men reference path should adopt the generalization-tuned Elo parameters as the active leader, while the women frozen leader remains `ARCH-04B`.

Re-test if:
- The Elo tuning objective seasons or play-in policy change again.
- Women Elo is tuned in full-blend context rather than standalone Elo context.
- A later men or women challenger beats these held-out values on the same 2023–2024 protocol.

Related:
- docs/adj_quality_gap_v10.md
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — VAL-05: women alpha-reopen reference challenger

Status: Completed

Hypothesis:
- Reopening the women Elo blend weight may recover complementary signal that the fixed `0.10` alpha is suppressing in the women reference-style model.

Dependencies:
- plan:002_gate2
- arch-04b
- feature:feat_13_women_hca_adj_qg_v1
- feature:feat_15_women_closewr3_v1
- validation:leave_season_out_v1

MLflow:
- Run name: val-05-women-alpha-reopen
- Run ID: 1ab9f3bd1cb549d0a0c7a3898bb2a385

Result:
- The re-estimated alpha selected `0.09` for the 2023 holdout, `0.27` for the 2024 holdout, and `0.27` for 2024 inference generation.
- On the 2023–2024 held-out window, the reopened-alpha women reference model scored `flat_brier = 0.137246` and `log_loss = 0.420522`.
- This improves over the fixed-`0.10` women reference runs (`0.138132`, `0.138383`, and `0.138507`) but still trails the frozen women leader `ARCH-04B` (`0.132193`).
- Conclusion: alpha re-estimation is the strongest women reference-path variant tested so far, but it still does not justify replacing `ARCH-04B`.

Re-test if:
- Women Elo is tuned jointly with alpha instead of through a two-stage procedure.
- The women feature set changes while keeping the current evaluation protocol fixed.
- A later women-only challenger approaches `ARCH-04B` closely enough to justify another blend-context search.

Related:
- docs/adj_quality_gap_v10.md
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — VAL-05: women reference-style replication with locally tuned Elo

Status: Completed

Hypothesis:
- Replacing the copied women Elo hyperparameters with locally tuned values may close the remaining women benchmark gap and improve the fuller women reference-style replication.

Dependencies:
- plan:002_gate2
- arch-04b
- feature:feat_13_women_hca_adj_qg_v1
- feature:feat_15_women_closewr3_v1
- feature:elo_tuned_carryover_women_local_v1
- validation:leave_season_out_v1

MLflow:
- Run name: val-05-women-reference-replication-v2-local-elo
- Run ID: a175a91ebe204a448b7895a8daf5e82d

Result:
- The local women Elo search selected `initial_rating=1254.7487`, `k_factor=93.9752`, `home_advantage=63.9890`, `season_carryover=0.96522`, `scale=2144.3093`, `mov_alpha=13.6431`, `weight_regular=1.7052`, `weight_tourney=0.7684`.
- On the 2023–2024 held-out window, the updated women reference model scored `flat_brier = 0.138383` and `log_loss = 0.422074`.
- This did not improve the earlier women reference replication (`0.138132`) and still trails the frozen women leader `ARCH-04B` (`0.132193`).
- Conclusion: local women Elo tuning does not currently justify changing the frozen women model. The women replication track should pause unless a narrower hypothesis emerges.

Re-test if:
- The women Elo tuning objective changes from Elo-only to full-model validation.
- The women blend weight is re-opened instead of fixed at `0.10`.
- Exact reference-model preprocessing details become available and suggest a concrete missing implementation detail.

Related:
- docs/adj_quality_gap_v10.md
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — VAL-01 / VAL-05: men reference-style margin regression with locally tuned Elo

Status: Completed

Hypothesis:
- Replacing the copied men Elo hyperparameters with locally tuned values can push the closer-match reference replication past the frozen men baseline without sacrificing the 2025 holdout.

Dependencies:
- plan:002_gate2
- arch-01
- feature:men_situational_v1
- feature:elo_tuned_carryover_men_local_v1
- validation:leave_season_out_v1

MLflow:
- Run name: val-05-men-reference-margin-v3-local-elo
- Run ID: f160c1daa6da4f94a1b3e9df6e2f308f
- Run name: val-01-2025-holdout-men-reference-v3
- Run ID: a11c60d33cde4fd68f7852fc65dda1db

Result:
- The locally tuned men Elo search selected `initial_rating=1636.4991`, `k_factor=52.6864`, `home_advantage=15.9927`, `season_carryover=0.95527`, `scale=1986.9244`, `mov_alpha=8.9837`, `weight_regular=1.7052`, `weight_tourney=1.0465`.
- On the 2023–2024 held-out window, the updated men reference model scored `flat_brier = 0.195985`, beating frozen `ARCH-01` (`0.197508`) and the prior closer-match replication (`0.197512`).
- The 2025 holdout sanity check also passed with `flat_brier = 0.129694` and `log_loss = 0.409406`, so the improvement is not confined to the original benchmark window.
- Conclusion: this is the first men challenger that cleanly displaces `ARCH-01`. The active men leader should move to the locally tuned reference-style margin model.

Re-test if:
- The Elo tuning search space or objective seasons change.
- Exact challenger-doc alpha/T implementation details become available.
- A later men model beats the tuned local-Elo reference path on held-out flat Brier.

Related:
- docs/adj_quality_gap_v10.md
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — VAL-05: men reference-style margin regression challenger (closer-match preprocessing)

Status: Completed

Hypothesis:
- Aligning the men reference runner more closely to the challenger document's exact preprocessing choices should materially improve the earlier replication attempt and may overtake the frozen men baseline.

Dependencies:
- plan:002_gate2
- arch-01
- feature:feat_12_iter_adj_qg_v1
- feature:men_situational_v1
- feature:elo_tuned_carryover_men_v1
- validation:leave_season_out_v1

MLflow:
- Run name: val-05-men-reference-margin-v2
- Run ID: db98ed1646cd400ead567ae4b6469a9a

Result:
- Men: `flat_brier = 0.197512` and `log_loss = 0.582183` on the 2023–2024 held-out window.
- This materially improves the earlier men reference run (`0.199918`) by correcting the challenger-specific preprocessing choices: `SeedDiff` sign, 2004 season floor, play-ins in training, drop/zero-impute policy, and men-specific `T`/`alpha` calibration.
- It is now effectively tied with frozen `ARCH-01` (`0.197508`) but still does not beat it on the repo's primary metric. The benchmark gap narrowed sharply, but the men leader should remain frozen for now.

Re-test if:
- Men Elo parameters are tuned locally instead of copied from the challenger document.
- The challenger's exact alpha/T estimation implementation details become available.
- A later men replication step adds any remaining preprocessing detail without degrading recent holdout seasons.

Related:
- docs/adj_quality_gap_v10.md
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — VAL-05: men reference-style margin regression challenger

Status: Completed

Hypothesis:
- Adding the challenger document's men-oriented situational features and switching to margin regression with Gaussian-CDF conversion will materially close the benchmark gap and potentially beat the frozen men baseline.

Dependencies:
- plan:002_gate2
- arch-01
- feature:feat_12_iter_adj_qg_v1
- feature:men_situational_v1
- feature:elo_tuned_carryover_men_v1
- validation:leave_season_out_v1

MLflow:
- Run name: val-05-men-reference-margin
- Run ID: 252ec46a6a874810b9e42237f7b23b1c

Result:
- Men: `flat_brier = 0.199918` and `log_loss = 0.586186` on the 2023–2024 held-out window.
- This improves on the broader adjusted-quality-gap challenger (`FEAT-12` at `0.200421`) and the earlier Phase A+B / tree challengers, but it still trails the frozen men leader `ARCH-01` (`0.197508`) and the reference benchmark trajectory.
- Round diagnostics were `r1_brier = 0.193375` and `r2plus_brier = 0.206672` across the two holdout seasons. The men reference path is now partially replicated, but not yet enough to displace the frozen baseline.

Re-test if:
- The men feature set is extended to match the reference document more exactly.
- The margin-to-probability conversion or residual-sigma estimation changes.
- A later challenger adds local Elo tuning or exact preprocessing conventions from the reference implementation.

Related:
- docs/adj_quality_gap_v10.md
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — VAL-05: frozen-model benchmark comparison vs `adj_quality_gap_v10`

Status: Completed

Hypothesis:
- The repo's frozen leaders can be compared fairly against the documented challenger on matched LOSO played-game Brier seasons, giving a clear benchmark gap by league.

Dependencies:
- plan:002_gate2
- arch-01
- arch-04b
- val-05
- validation:leave_season_out_v1

MLflow:
- Run name: val-05-men-frozen-loso-v2
- Run ID: 4a44ac9fdc8345e1aaf0ce56d19b70c8
- Run name: val-05-women-frozen-loso
- Run ID: a8ff2953ecb0446689d77e15d541f95b

Result:
- Men frozen leader (locally tuned reference margin model) averaged `0.184261` across the same LOSO seasons reported in the challenger doc, versus the reference mean `0.185340`. Men now slightly beat the benchmark on matched LOSO mean.
- Women frozen leader (`ARCH-04B`) averaged `0.138636` versus the reference mean `0.130480`. Women are more competitive, especially in recent seasons, but still below the full-history benchmark.
- Season-by-season, the updated men leader beats the reference in `2010`, `2013`, `2014`, `2015`, `2016`, `2017`, `2019`, `2024`, and `2025`, while the women leader beats it in `2013`, `2019`, `2021`, `2023`, and `2024`.
- This is now an apples-to-apples benchmark comparison on played-game LOSO Brier by season. It is still not an exact replication comparison because the underlying model families and training setups differ.

Re-test if:
- The frozen leaders change.
- The benchmark document's season table is corrected or revised.
- A later local challenger materially closes the benchmark gap on either league.

Related:
- docs/adj_quality_gap_v10.md
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — VAL-01: 2025 holdout sanity check (women, ARCH-04B)

Status: Completed

Hypothesis:
- The tuned-Elo women baseline (`ARCH-04B`) generalizes from the 2023–2024 validation window to the unseen 2025 tournament without a material degradation.

Dependencies:
- plan:002_gate2
- arch-04b
- validation:leave_season_out_v1

MLflow:
- Run name: val-01-2025-holdout-women-arch-04b
- Run ID: f6eac553495949a5805c84fb4cdef757

Result:
- 2025 holdout produced `flat_brier = 0.107988` and `log_loss = 0.341670`.
- Mean round-group diagnostics: `r1_brier = 0.086122`, `r2plus_brier = 0.130561`.
- This is materially better than the earlier 2023–2024 validation average for `ARCH-04B` (`0.132193`), so there is no overfitting signal in the tuned-Elo replacement. The women leader is ready to freeze for Gate 3.

Re-test if:
- The tuned Elo parameters or carryover policy change.
- A later women challenger beats `ARCH-04B` on held-out flat Brier.
- Submission-time feature generation diverges from the current tuned-Elo baseline path.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — ARCH-04B: tuned carryover Elo + seed logistic baseline (women)

Status: Completed

Hypothesis:
- Replacing the simple Elo engine in `ARCH-04` with the reference-style tuned carryover Elo will improve the women’s seed-plus-Elo baseline without adding extra feature complexity.

Dependencies:
- plan:002_gate2
- arch-04
- feature:seed_diff_v1
- feature:elo_tuned_carryover_women_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-04b-tuned-elo-seed-women
- Run ID: ba0a02b8c8bc404089bdd2de7fe9a917

Result:
- Women: `flat_brier = 0.132193`, compared with raw `ARCH-04` at `0.137357`.
- This is also better than the fuller women reference replication (`0.138132`) and all earlier women challengers in the current branch.
- Conclusion: the tuned/carryover Elo replacement is the first challenger to beat the standing women leader on the repo’s primary metric. It becomes the new active women baseline pending any later challenger.

Re-test if:
- The tuned Elo parameters are re-fit locally instead of copied from the reference document.
- Tournament-game carryover into future seasons is disabled or reweighted.
- A later women model beats `0.132193` on the same held-out flat-Brier protocol.

Related:
- docs/adj_quality_gap_v10.md
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — VAL-05: women reference-style replication with tuned carryover Elo

Status: Completed

Hypothesis:
- A fuller women reference-style model using tuned carryover Elo, HCA-adjusted quality gap, CloseWR3, and a fixed 10% Elo blend may beat the current frozen women baseline and approach the documented benchmark.

Dependencies:
- plan:002_gate2
- arch-04
- feature:feat_13_women_hca_adj_qg_v1
- feature:feat_15_women_closewr3_v1
- feature:elo_tuned_carryover_women_v1
- validation:leave_season_out_v1

MLflow:
- Run name: val-05-women-reference-replication
- Run ID: 44361f54bf444ba5a906238538af0cba

Result:
- Women: `flat_brier = 0.138132`, improving substantially over `FEAT-12` (`0.140347`), `FEAT-13` (`0.141027`), and `FEAT-15` (`0.141104`).
- This is still slightly worse than the frozen leader `ARCH-04` (`0.137357`) on the repo's primary metric, so it does not displace the current women baseline.
- Relative to the documented reference table, it is competitive on the 2023–2024 window and materially closer to the target benchmark than the earlier isolated challenger features. The tuned/carryover Elo plus fuller women feature bundle appears to be the first challenger path worth keeping active.

Re-test if:
- The women reference replication switches from the current logistic-plus-fixed-blend implementation to the exact documented preprocessing and training stack.
- The tuned Elo parameters are re-fit locally instead of copied from the reference document.
- A combined challenger adds the remaining reference-model choices without degrading flat Brier.

Related:
- docs/adj_quality_gap_v10.md
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — FEAT-15: women CloseWR3 challenger

Status: Completed

Hypothesis:
- Women's close-game win rate at the 3-point threshold may add signal beyond seed, Elo, and the HCA-adjusted quality gap.

Dependencies:
- plan:002_gate2
- arch-04
- feature:elo_v1
- feature:feat_13_women_hca_adj_qg_v1
- feature:feat_15_women_closewr3_v1
- validation:leave_season_out_v1

MLflow:
- Run name: feat-15-women-closewr3
- Run ID: f57fb223f785496b8fa80c259be57115

Result:
- Women: `flat_brier = 0.141104`, compared with `FEAT-13` at `0.141027`, `FEAT-12` at `0.140347`, and the frozen leader `ARCH-04` at `0.137357`.
- The `CloseWR3` addition produced only marginal movement and did not improve either holdout season enough to offset the extra feature complexity.
- Conclusion: the first `FEAT-15` implementation does not improve on the current women challengers or the frozen baseline.

Re-test if:
- `CloseWR3` is paired with the reference model's tuned Elo and fixed 10% Elo blend.
- The close-game feature is regularized or transformed instead of used as a raw season win rate.
- A fuller women challenger bundles `CloseWR3` with other reference-model choices rather than testing it in isolation.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — FEAT-13: women HCA-corrected adjusted quality gap challenger

Status: Completed

Hypothesis:
- Applying the documented women-only home-court correction before the SOS iteration will improve the adjusted quality-gap signal over the raw iterative version.

Dependencies:
- plan:002_gate2
- arch-04
- feature:elo_v1
- feature:feat_12_iter_adj_qg_v1
- feature:feat_13_women_hca_adj_qg_v1
- validation:leave_season_out_v1

MLflow:
- Run name: feat-13-women-hca-adj-qg
- Run ID: 674cc9811ca148efb72b9dbeaa799076

Result:
- Women: `flat_brier = 0.141027`, compared with `FEAT-12` women at `0.140347` and the frozen leader `ARCH-04` at `0.137357`.
- Round diagnostics remained close to FEAT-12, but the HCA-corrected version was slightly worse in both 2023 and 2024 holdouts.
- Conclusion: the first `FEAT-13` implementation does not improve on FEAT-12 or the frozen women’s baseline, so the women HCA correction should remain a documented challenger path rather than the active leader.

Re-test if:
- The women HCA magnitude is tuned rather than fixed at `3.0`.
- The HCA-corrected quality gap is paired with the richer women feature set from `FEAT-15`.
- Tuned or carryover Elo is added, making the combined challenger closer to the documented reference model.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-12 — FEAT-12: iterative adjusted quality gap challenger

Status: Completed

Hypothesis:
- KenPom-style iterative opponent-adjusted efficiency, combined with Elo and seed difference, may outperform the simpler SOS-adjusted proxy and beat the current frozen baselines.

Dependencies:
- plan:002_gate2
- arch-01
- arch-04
- feature:elo_v1
- feature:feat_12_iter_adj_qg_v1
- validation:leave_season_out_v1

MLflow:
- Run name: feat-12-iter-adj-qg-men
- Run ID: 54e0cb19318840aa9fdf01b29c4fc94c
- Run name: feat-12-iter-adj-qg-women
- Run ID: 5279d3fd507544898087160d521463fc

Result:
- Men: `flat_brier = 0.200421`, which improves on `FEAT-11` (`0.204907`) but remains worse than both `ARCH-01` (`0.197508`) and `ARCH-03` (`0.197613`).
- Women: `flat_brier = 0.140347`, which improves materially on `FEAT-11` (`0.146514`) and nearly matches `ARCH-02` (`0.140573`), but still trails the frozen leader `ARCH-04` (`0.137357`).
- Conclusion: the first `FEAT-12` implementation is directionally better than the simpler SOS-adjusted proxy, especially for women, but it still does not beat the frozen leaders on the primary metric.

Re-test if:
- The iterative adjustment formula changes to include home-court correction or richer box-score inputs.
- The challenger is paired with tuned Elo rather than the current baseline Elo implementation.
- Women-specific challenger work adds the planned HCA correction from `FEAT-13`.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — FEAT-11: SOS-adjusted net efficiency challenger

Status: Completed

Hypothesis:
- A season-normalized SOS-adjusted net-efficiency signal, combined with Elo, may outperform the current raw baselines by capturing team quality after accounting for schedule strength.

Dependencies:
- plan:002_gate2
- arch-03
- arch-04
- feature:elo_v1
- feature:feat_11_sos_adj_net_eff_v1
- validation:leave_season_out_v1

MLflow:
- Run name: feat-11-sos-adj-net-eff-men
- Run ID: d06d9567991346639c1ce31c2e22502d
- Run name: feat-11-sos-adj-net-eff-women
- Run ID: 922d8f5438c341f985e7c692d423f43a

Result:
- Men: `flat_brier = 0.204907`, which is materially worse than both `ARCH-01` (`0.197508`) and `ARCH-03` (`0.197613`).
- Women: `flat_brier = 0.146514`, which is worse than both `ARCH-02` (`0.140573`) and `ARCH-04` (`0.137357`).
- Conclusion: the first `FEAT-11` implementation does not improve on the frozen leaders, though the feature is now cleanly pre-registered and available for narrower future challengers.

Re-test if:
- The SOS-adjustment formula changes from the current season-normalized index definition.
- `net_eff` is used in a more targeted women-only or round-group-specific challenger.
- A later stack uses this feature as an upstream model component rather than a direct logistic term.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — Gate 2 consolidation: preserve raw baselines as current leaders

Status: Completed

Hypothesis:
- After completing the planned Gate 2 model family comparisons, the disciplined next step is to freeze the current candidate set to the best empirical models rather than continue expanding complexity without evidence.

Dependencies:
- plan:002_gate2
- arch-01
- arch-04
- combo-01
- combo-02
- val-02

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Gate 2 comparisons are now broad enough to support a candidate freeze: richer linear models, tree models, XGBoost retries, calibration audits, and simple blends all failed to beat the raw baselines on flat Brier.
- Current leading candidates are raw `ARCH-01` for men and raw `ARCH-04` for women.
- Future work should be treated as challenger experiments only; no model should replace the current leaders without beating them on held-out flat Brier.

Re-test if:
- A stacked or round-group ensemble beats the raw leaders on flat Brier.
- A narrower feature experiment produces a measurable improvement over the raw baselines.
- The 2025 holdout or later season checks contradict the current ranking.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — COMBO-01 and COMBO-02: simple blend of baseline and tree model

Status: Completed

Hypothesis:
- A simple weighted blend of the strongest linear baseline and the Phase A+B+C tree model may improve held-out tournament Brier over either model alone.

Dependencies:
- plan:002_gate2
- arch-03
- arch-04
- arch-07
- arch-08
- validation:leave_season_out_v1

MLflow:
- Run name: combo-01-simple-blend-men
- Run ID: 9b313ac531e044a182029a82f3fef182
- Run name: combo-02-simple-blend-women
- Run ID: d13e89d9a38448f9ab01dd76e20fc879

Result:
- `COMBO-01` men produced `flat_brier = 0.198945` and `log_loss = 0.582452`, with the OOF-selected weight staying at `0.95` on the Elo logistic component in both holdout seasons. This underperformed both `ARCH-01` (`0.197508`) and `ARCH-03` (`0.197613`).
- `COMBO-02` women produced `flat_brier = 0.137713` and `log_loss = 0.425836`, with OOF-selected weights of `0.70` and `0.75` on the Elo logistic component. This was extremely close to `ARCH-04` (`0.137357`) but still slightly worse on the primary metric.
- Conclusion: neither simple blend displaces the current best raw baselines, though the women’s blend suggests the tree model may carry a small amount of complementary signal.

Re-test if:
- A later stack or calibrated ensemble uses the same members with a different blending policy.
- Women’s tree model improves enough to make the simple blend clearly positive.
- Round-group models become available as alternate blend members.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — VAL-02: calibration audit for current best baselines

Status: Completed

Hypothesis:
- Training-fold calibration (isotonic or Platt) may improve held-out tournament Brier for the strongest current baseline candidates.

Dependencies:
- plan:002_gate2
- arch-01
- arch-03
- arch-04
- validation:leave_season_out_v1

MLflow:
- Run name: val-02-arch-01-men
- Run ID: bcfa7d9e02e04600a95f214fbd1c2fd5
- Run name: val-02-arch-03-men
- Run ID: 28a78f11880b42d88c6dae21b07ee7f6
- Run name: val-02-arch-04-women
- Run ID: 553e266635914bcc832d3efb94538623

Result:
- `ARCH-01` men: raw remained best (`flat_brier = 0.197508`) over Platt (`0.201503`) and isotonic (`0.203171`).
- `ARCH-03` men: raw remained best (`flat_brier = 0.197613`), though isotonic improved ECE (`0.108522` vs raw `0.138388`) at a small Brier cost (`0.199500`).
- `ARCH-04` women: raw remained best (`flat_brier = 0.137357`); isotonic improved ECE (`0.064986` vs raw `0.090089`) but materially hurt Brier (`0.144300`).
- Conclusion: VAL-02 is complete for the current best candidates, and no calibrated variant displaces the raw baseline on the primary selection metric.

Re-test if:
- Ensemble or stacked outputs become the new calibration target.
- A later model family is materially less calibrated than the current baselines.
- Submission-time clipping policy changes and affects calibrated tails.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-07 retry: XGBoost on Phase A+B+C features (men)

Status: Completed

Hypothesis:
- Replacing the histogram GBT surrogate with true XGBoost will improve men’s tree-model performance on the Phase A+B+C feature set.

Dependencies:
- plan:002_gate2
- arch-07
- feature:phase_c_v1
- validation:leave_season_out_v1
- dependency:xgboost

MLflow:
- Run name: arch-07-xgboost-men
- Run ID: 087a77668be3416b99645578b2978e7a

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.202180` and `log_loss = 0.579151`.
- Mean round-group diagnostics: `r1_brier = 0.192497`, `r2plus_brier = 0.212175`.
- This was slightly worse than the earlier histogram GBT run (`ARCH-07`, `flat_brier = 0.200974`) and still worse than the current men’s baselines, so XGBoost does not advance the candidate set on the current feature family.

Re-test if:
- XGBoost hyperparameters are tuned rather than using the current fixed starter configuration.
- Phase C is expanded with stronger matchup priors.
- A calibrated XGBoost layer is tested later in Gate 2 or Gate 3.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-08 retry: XGBoost on Phase A+B+C features (women)

Status: Completed

Hypothesis:
- True XGBoost will improve women’s tree-model performance on the Phase A+B+C feature set relative to the earlier histogram GBT attempt.

Dependencies:
- plan:002_gate2
- arch-08
- feature:phase_c_v1
- validation:leave_season_out_v1
- dependency:xgboost

MLflow:
- Run name: arch-08-xgboost-women
- Run ID: bed6b72c7159465282218c492713bd6c

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.172717` and `log_loss = 0.510730`.
- Mean round-group diagnostics: `r1_brier = 0.137896`, `r2plus_brier = 0.208661`.
- This was materially worse than the earlier histogram GBT run (`ARCH-08`, `flat_brier = 0.155726`) and far behind the current women’s baseline `ARCH-04`, so XGBoost does not advance the candidate set on the current feature family.

Re-test if:
- The women’s XGBoost feature set is narrowed or regularized.
- A different season weighting or hyperparameter search is introduced.
- Later experiments add stronger contextual features than the current Phase C subset.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-07: Phase A+B+C GBT primary model (men)

Status: Completed

Hypothesis:
- A tree model on men’s Phase A+B+C features will capture nonlinear interactions that the logistic baselines miss and improve held-out tournament Brier.

Dependencies:
- plan:002_gate2
- arch-03
- arch-05
- feature:phase_b_v1
- feature:phase_c_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-07-phase-abc-gbt-men
- Run ID: 0cfe2f69719944bcb79502464af946b6

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.200974` and `log_loss = 0.573329`.
- Mean round-group diagnostics: `r1_brier = 0.188467`, `r2plus_brier = 0.213885`.
- This improved over `ARCH-05` (`flat_brier = 0.203797`) but still underperformed both `ARCH-01` (`0.197508`) and `ARCH-03` (`0.197613`), so it does not displace the current men’s baseline.

Re-test if:
- Phase C is expanded beyond the current efficiency and tempo interactions.
- Tree hyperparameters are tuned or calibration is added.
- Seed-pair historical priors are added as a separate feature family.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-08: Phase A+B+C GBT primary model (women)

Status: Completed

Hypothesis:
- A tree model on women’s Phase A+B+C features will capture nonlinear interactions and outperform the current linear baselines.

Dependencies:
- plan:002_gate2
- arch-04
- arch-06
- feature:phase_b_v1
- feature:phase_c_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-08-phase-abc-gbt-women
- Run ID: 98c0ca0b08c246409b782aaf627d7641

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.155726` and `log_loss = 0.463890`.
- Mean round-group diagnostics: `r1_brier = 0.126368`, `r2plus_brier = 0.186031`.
- This was worse than `ARCH-06` (`flat_brier = 0.143717`) and materially worse than the current best women’s baseline `ARCH-04` (`0.137357`), so it should not advance as a candidate.

Re-test if:
- The women’s tree model gets a narrower, better-regularized feature set.
- Calibration is added after the tree model.
- Later Gate 2 work adds stronger matchup/context features than the current Phase C subset.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-05: Phase A+B logistic primary model (men)

Status: Completed

Hypothesis:
- A richer men’s Phase A+B logistic model will improve tournament Brier over the seed-only and seed-plus-Elo baselines.

Dependencies:
- plan:002_gate2
- arch-03
- feature:phase_b_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-05-phase-ab-men
- Run ID: ffd0b86de56e40b7ac6a13e4990420c1

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.203797` and `log_loss = 0.599488`.
- Mean round-group diagnostics: `r1_brier = 0.200083`, `r2plus_brier = 0.207631`.
- This underperformed both `ARCH-01` (`flat_brier = 0.197508`) and `ARCH-03` (`flat_brier = 0.197613`), so the first men’s primary-model attempt should not advance as a candidate.

Re-test if:
- The men’s Phase A+B feature set is pruned or regularized before rerunning.
- Matchup-level Phase C interactions are added.
- Calibration is added on top of the men’s primary model family.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-06: Phase A+B logistic primary model (women)

Status: Completed

Hypothesis:
- A richer women’s Phase A+B logistic model will improve tournament Brier over the current Elo-plus-seed baseline.

Dependencies:
- plan:002_gate2
- arch-04
- feature:phase_b_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-06-phase-ab-women
- Run ID: d7fa5c4793f4497ba8bb7d135591159b

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.143717` and `log_loss = 0.436281`.
- Mean round-group diagnostics: `r1_brier = 0.109516`, `r2plus_brier = 0.179020`.
- This underperformed both `ARCH-02` (`flat_brier = 0.140573`) and `ARCH-04` (`flat_brier = 0.137357`), so the first women’s primary-model attempt should not advance as a candidate.

Re-test if:
- The women’s Phase A+B feature set is simplified or reweighted before rerunning.
- Matchup-level Phase C interactions are added.
- A tree-based model is used instead of linear logistic regression on the same feature family.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — Gate 1: Feature Phase B builders implemented

Status: Completed

Hypothesis:
- Reusable Phase B season-level feature builders will let primary-model experiments move beyond seed and Elo baselines without forcing feature logic to live in notebooks or one-off scripts.

Dependencies:
- plan:002_gate1
- feature:elo_v1
- round_assignment:seed_pair_lookup

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added `src/mmlm2026/features/phase_b.py` with season-level builders for adjusted efficiency, strength of schedule, recent form, and Massey consensus features.
- Added `tests/test_phase_b_features.py`; `uv run pytest`, `uv run ruff check .`, and `uv run mypy src` passed locally after the Phase B implementation.
- Gate 1 feature infrastructure now covers the Phase B feature families called for in PLAN-002, so the next work can move to primary-model runs instead of additional baseline plumbing.

Re-test if:
- The pre-tournament cutoff policy changes away from `DayNum < 134` / `RankingDayNum <= 133`.
- Phase B feature definitions change materially before ARCH-05 and ARCH-06.
- A selected model family needs matchup-level transformations beyond the current season-level builders.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-09 — Infrastructure complete; no model experiments run yet

Status: Completed

Hypothesis:
- N/A — this is an infrastructure milestone marker, not a model experiment.

Dependencies:
- N/A

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Phases 1-12 of the original setup plan are complete (see `docs/roadmaps/EXECUTION_STATUS.md`).
- Stage 1 infrastructure implemented: data loaders (`src/mmlm2026/data/loaders.py`),
  evaluation splits (`src/mmlm2026/evaluation/splits.py`), submission writer
  (`src/mmlm2026/submission/writer.py`), MLflow helper
  (`src/mmlm2026/utils/mlflow_tracking.py`), round assignment utility
  (`src/mmlm2026/round_utils.py`).
- No predictive model experiments have been run as of this date.

Re-test if:
- N/A

Related:
- docs/roadmaps/EXECUTION_STATUS.md
- docs/decisions/0001-experiment-ledger-is-mlflow.md

---
## 2026-03-10 — Gate 0: submission validation utility implemented

Status: Completed

Hypothesis:
- A dedicated submission validator will catch schema and ID issues before upload and reduce last-minute submission failures.

Dependencies:
- plan:002_gate0
- submission_writer:v1

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added `src/mmlm2026/submission/validation.py` with reusable validation for schema, ID format, duplicate IDs, prediction range, and optional sample-submission coverage checks.
- Added `scripts/validate_submission.py` as the Gate 0 CLI entry point referenced by PLAN-002.
- Added tests in `tests/test_submission_validation.py`; `uv run pytest` passed locally.

Re-test if:
- Kaggle submission schema changes.
- The project changes clipping bounds away from `[0.025, 0.975]`.
- Sample-submission handling becomes league- or stage-specific.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — VAL-01: 2025 holdout sanity check (men)

Status: Completed

Hypothesis:
- The current best men’s baseline (`ARCH-01`) generalizes from the 2023–2024 validation window to the unseen 2025 tournament without a material degradation.

Dependencies:
- plan:002_gate1
- arch-01
- validation:leave_season_out_v1

MLflow:
- Run name: val-01-2025-holdout-men
- Run ID: c3733a50efc7491586f3060e0dc317b0

Result:
- 2025 holdout produced `flat_brier = 0.151642` and `log_loss = 0.469892`.
- Mean round-group diagnostics: `r1_brier = 0.135707`, `r2plus_brier = 0.168092`.
- This is better than the earlier 2023–2024 validation average from `ARCH-01` (`flat_brier = 0.197508`), so there is no evidence of overfitting from the current men’s baseline.

Re-test if:
- A new men’s baseline overtakes `ARCH-01`.
- The validation window or leakage policy changes.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — VAL-01: 2025 holdout sanity check (women)

Status: Completed

Hypothesis:
- The current best women’s baseline (`ARCH-04`) generalizes from the 2023–2024 validation window to the unseen 2025 tournament without a material degradation.

Dependencies:
- plan:002_gate1
- arch-04
- validation:leave_season_out_v1

MLflow:
- Run name: val-01-2025-holdout-women
- Run ID: 84f848e9634340f1adf0233d36ded46f

Result:
- 2025 holdout produced `flat_brier = 0.113286` and `log_loss = 0.361987`.
- Mean round-group diagnostics: `r1_brier = 0.087981`, `r2plus_brier = 0.139406`.
- This is better than the earlier 2023–2024 validation average from `ARCH-04` (`flat_brier = 0.137357`), so there is no evidence of overfitting from the current women’s baseline.

Re-test if:
- A new women’s baseline overtakes `ARCH-04`.
- The validation window or leakage policy changes.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-03: Elo + seed logistic baseline (men)

Status: Completed

Hypothesis:
- End-of-regular-season Elo adds signal beyond seed difference for the men’s tournament baseline.

Dependencies:
- plan:002_gate1
- feature:seed_diff_v1
- feature:elo_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-03-elo-seed-men
- Run ID: ed65e8bcc7e04961a97bbccd4e868c71

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.197613` and `log_loss = 0.580609`.
- Mean round-group diagnostics: `r1_brier = 0.185721`, `r2plus_brier = 0.209889`.
- Relative to `ARCH-01` (`flat_brier = 0.197508`), Elo did not improve the men’s baseline on this validation window.

Re-test if:
- Elo hyperparameters (`day_cutoff`, `k_factor`, `home_advantage`) change.
- Men’s baseline feature set expands beyond `seed_diff` and `elo_diff`.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-04: Elo + seed logistic baseline (women)

Status: Completed

Hypothesis:
- End-of-regular-season Elo adds signal beyond seed difference for the women’s tournament baseline.

Dependencies:
- plan:002_gate1
- feature:seed_diff_v1
- feature:elo_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-04-elo-seed-women
- Run ID: c4929b684eea490faaca353aefbb5e46

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.137357` and `log_loss = 0.426218`.
- Mean round-group diagnostics: `r1_brier = 0.103508`, `r2plus_brier = 0.172297`.
- Relative to `ARCH-02` (`flat_brier = 0.140573`), Elo improved the women’s baseline on this validation window.

Re-test if:
- Elo hyperparameters (`day_cutoff`, `k_factor`, `home_advantage`) change.
- Women’s baseline feature set expands beyond `seed_diff` and `elo_diff`.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-01: seed-diff logistic baseline (men)

Status: Completed

Hypothesis:
- Seed difference alone is a strong men’s tournament baseline and should beat the naive 0.5 floor with reasonable Round of 64 calibration.

Dependencies:
- plan:002_gate1
- feature:seed_diff_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-01-seed-diff-men
- Run ID: c0ef5d21cbf44cc185edc4ec7b920b43

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.197508` and `log_loss = 0.579247`.
- Mean round-group diagnostics: `r1_brier = 0.182200`, `r2plus_brier = 0.213310`.
- Per-season flat Brier: 2023 `0.204203`, 2024 `0.190814`.

Re-test if:
- Seed parsing or `seed_diff` sign convention changes.
- Round-group routing or bracket diagnostics change materially.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — ARCH-02: seed-diff logistic baseline (women)

Status: Completed

Hypothesis:
- Seed difference alone is an even stronger women’s tournament baseline than men’s and should provide a solid floor for later feature additions.

Dependencies:
- plan:002_gate1
- feature:seed_diff_v1
- validation:leave_season_out_v1

MLflow:
- Run name: arch-02-seed-diff-women
- Run ID: 809fa43caaf24469b2a2b68cf3afbfc9

Result:
- Leave-seasons-out validation on 2023 and 2024 produced overall `flat_brier = 0.140573` and `log_loss = 0.432945`.
- Mean round-group diagnostics: `r1_brier = 0.104118`, `r2plus_brier = 0.178204`.
- Per-season flat Brier: 2023 `0.169764`, 2024 `0.111382`.

Re-test if:
- Seed parsing or `seed_diff` sign convention changes.
- Women’s tournament seed dynamics shift materially in future seasons.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — Gate 0: seed-diff baseline run plumbing implemented

Status: Completed

Hypothesis:
- A dedicated seed-diff baseline runner will make ARCH-01 and ARCH-02 reproducible and close the remaining Gate 0 plumbing gap before actual MLflow baseline runs.

Dependencies:
- plan:002_gate0
- validate_cv:v1
- bracket_dp:v1

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added `src/mmlm2026/features/baseline.py` to build played-game seed-diff tournament features in canonical `LowTeamID` orientation with `round_group` and `outcome`.
- Added `scripts/run_seed_diff_baseline.py` to build baseline features from raw Kaggle tournament files, run leave-season-out validation, generate bracket diagnostics for the latest holdout season, and optionally log to MLflow.
- Added `tests/test_baseline_features.py`; `uv run pytest`, `uv run ruff check .`, and `uv run mypy src` passed locally.

Re-test if:
- Seed-diff feature orientation or sign convention changes.
- ARCH-01/02 move to a different baseline model family.
- Stage 2 baseline inference requires additional fields beyond the current canonical baseline feature set.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — Gate 0: bracket diagnostics utility implemented

Status: Completed

Hypothesis:
- Deterministic bracket DP diagnostics will expose which matchup probabilities matter most in bracket space without changing the primary flat-Brier selection metric.

Dependencies:
- plan:002_gate0
- validate_cv:v1
- round_assignment:seed_pair_lookup

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added `src/mmlm2026/evaluation/bracket.py` with slot recursion, play-in slot support, per-pair play probabilities, per-slot win probabilities, and optional realized-game bucket/round summaries.
- Added `scripts/bracket_dp.py` as the CLI for season-specific bracket diagnostics from slots, seeds, matchup probabilities, and optional tournament results.
- Added tests in `tests/test_bracket.py`; `uv run pytest`, `uv run ruff check .`, and `uv run mypy src` passed locally.

Re-test if:
- Kaggle slot graph conventions change.
- Play-in slot handling changes for men or women.
- The project changes bucket thresholds or diagnostic summary expectations.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md

---
## 2026-03-10 — Gate 0: leave-season-out CV validation utility implemented

Status: Completed

Hypothesis:
- A reusable leave-season-out validation CLI will make baseline and model-family comparisons reproducible and enforce the plan's temporal evaluation rules.

Dependencies:
- plan:002_gate0
- submission_validation:v1
- mlflow_tracking:v1

MLflow:
- Run name: N/A
- Run ID: N/A

Result:
- Added `src/mmlm2026/evaluation/validation.py` with strict pre-holdout training, per-season flat Brier/log-loss, optional `R1` and `R2+` diagnostics, calibration-table generation, and reliability-diagram export.
- Added `scripts/validate_cv.py` as the Gate 0 CLI entry point with optional MLflow logging and required `leakage_audit` tagging when MLflow logging is enabled.
- Added tests in `tests/test_validation.py`; `uv run pytest`, `uv run ruff check .`, and `uv run mypy src` passed locally. `scikit-learn` imports are locally documented with `import-untyped` suppressions because the package is untyped in this environment.

Re-test if:
- The validation split policy changes away from strict pre-holdout training.
- The repo adopts a different baseline model family or calibration workflow.
- `scikit-learn` ships typed packages and the local `import-untyped` suppressions can be removed.

Related:
- docs/roadmaps/PLAN-002-competition-attack-plan.md
- docs/roadmaps/EXECUTION_STATUS.md
