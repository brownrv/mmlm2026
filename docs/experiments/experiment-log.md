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
