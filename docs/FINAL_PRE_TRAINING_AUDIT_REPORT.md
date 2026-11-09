# Comprehensive Final Pre-Training Audit Report

**Project Title:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Review Role:** Independent Senior ML Engineer, Power-Systems Data Scientist, Statistical Auditor & Research Methodology Reviewer  
**Audit Timestamp:** 2026-11-02 13:10 IST  
**Overall Decision:** **GO — All Critical Checks Passed; Training is Authorized**  
**Audit Checklist CSV:** [`docs/FINAL_PRE_TRAINING_AUDIT_CHECKLIST.csv`](./FINAL_PRE_TRAINING_AUDIT_CHECKLIST.csv)  
**Blockers Register:** [`docs/FINAL_PRE_TRAINING_BLOCKERS.md`](./FINAL_PRE_TRAINING_BLOCKERS.md)  
**Summary JSON:** [`artifacts/metrics/pre_training_audit_summary.json`](../artifacts/metrics/pre_training_audit_summary.json)  
**Automated Unit Tests:** **29 / 29 PASS (100.0%)** (`python -m unittest discover tests`)

---

## Executive Summary

An exhaustive, multi-disciplinary pre-training audit of the entire end-to-end repository was conducted prior to authorizing deep learning model training. The audit independently inspected raw datasets, manifests, target construction routines, meteorological features, exploratory figures, sequence extraction and boundary purge algorithms, scaling mechanics, baseline benchmark calculations, model architecture definitions, and execution pipelines.

### Key Audit Findings:
1. **Raw Telemetry & Immutability:** All 9 raw files (1,077,613,531 bytes) and 6 raw weather files were independently hashed. 100% of SHA-256 and MD5 hashes match acquisition manifests exactly. Zero raw bytes have been modified or overwritten.
2. **Target & Operating State Rigor:** IIIT Delhi target demand is correctly reconstructed using physical equations ($\text{kWh} = \exp(\text{lnenergy})$, average $\text{kW} = 4 \times \text{kWh}$). CEEW 15-minute targets enforce the strict 5-subinterval set $\mathcal{S}(T) = \{T-12, T-9, T-6, T-3, T\}$; incomplete windows ($0.13\%\text{--}0.31\%$) are strictly rejected to NaN. Consumer zero load ($V \ge 150\text{V}, I = 0\text{A}$) and grid blackouts ($V < 50\text{V}, I = 0\text{A}$) are preserved and disaggregated.
3. **Meteorological Covariate Separation:** Dual weather matrices maintain complete structural separation between retrospective reanalysis (`retro_*`) and operational availability with 5-day publication latency (`operational_*`, $t_{\text{weather}} \le T - 120\text{h}$). All merges use causal backward as-of alignment (`direction="backward"`).
4. **EDA Validity & Domain Bounding:** The Bareilly household panel is fully populated using genuinely observed telemetry from `BR02`–`BR06` ($672 / 672$ non-null observations; $284,206$ colored pixels verified). Mathura's temperature-demand quadratic regression curve is strictly bounded to the observed empirical temperature domain ($[8.7, 43.8]^\circ\text{C}$) with `truncate=True` and non-negative demand bounding (`ylim(bottom=0)`). Pearson and Spearman statistics are reported on full valid pairwise observations, and regression curves are explicitly labeled as exploratory associations.
5. **Sequence Census & Boundary Purge:** Sequence yield across all 103 individual meters was reconciled to the exact ground truth: **4,467,924 contiguous sequences** ($L=16, H=1$) with **exact 0 discrepancy** across Train (70%), Purge 1 (16 steps), Val (15%), Purge 2 (16 steps), and Test partitions. The 16-step sequence-window purge was audited across all 106 series (`docs/all_series_boundary_purge_audit.csv`), confirming **zero cross-partition timestamp overlap ($\emptyset$)** and a mathematically exact 15.0-minute boundary gap.
6. **Model Parity & Pipeline Hygiene:** 1D CNN (~52k-54k params), LSTM (~52k-55k params), and CNN-LSTM (~69k-71k params) exhibit balanced parameter capacity. Scalers are fit strictly on training partitions only. The Google Colab pipeline notebook was updated with environment checks, Google Drive mounting instructions, and resilient path resolution.

---

## Detailed Audit Modules (1 to 10)

```
========================================================================================================================
AUDIT 1 — RAW DATA AND PROVENANCE
========================================================================================================================
Check 1.1: Source & Licensing
  - Dataset A: IIIT Delhi Faculty Housing Dataset. DOI: 10.7910/DVN/7MEXN4. License: CC0 1.0 Universal.
  - Dataset B: CEEW Smart Meter Data (Mathura & Bareilly). DOI: 10.7910/DVN/GOCHJH. License: CC BY-NC-SA 4.0.
  - Status: PASS.
Check 1.2: File Hashes & Immutability
  - Table_A2_Final.tab (30,494,398 bytes): SHA-256 matching 0833a466827a...
  - CEEW Bareilly 2019-2021 (661,543,624 bytes total): All 3 annual files hash verified.
  - CEEW Mathura 2019-2021 (385,381,157 bytes total): All 3 annual files hash verified.
  - CEEW ReadMe & Delhi PDF documentation: Hash verified.
  - Read-only attribute verified. Zero bytes modified.
  - Status: PASS.
Check 1.3: Cadence, Timezone & Conventions
  - All timestamps standardized to Indian Standard Time (IST, UTC+5:30).
  - Native cadence verified: 3-minute for CEEW, 15-minute for IIIT Delhi.
  - Interval-ending closing timestamp convention enforced across both cohorts.
  - Status: PASS.
Check 1.4: Longitudinal Cohorts & Boundary Transitions
  - IIIT Delhi: 19 faculty apartments.
  - CEEW Mathura: 38 fixed longitudinal smart meters (>=50,000 packets spanning 2019 to 2020/2021).
  - CEEW Bareilly: 46 fixed longitudinal smart meters (>=50,000 packets spanning 2019 to 2020/2021).
  - New Year transition gaps characterized in audit report; no synthetic bridging across midnight.
  - Status: PASS.
Check 1.5: Non-Fabrication Verification
  - Zero synthetic data points generated. Zero forward/backward filling across missing telemetry.
  - Status: PASS.

========================================================================================================================
AUDIT 2 — TARGET AND FEATURE CONSTRUCTION
========================================================================================================================
Check 2.1: IIIT Delhi Target Construction
  - Energy recovery formula: energy_15min_kwh = exp(lnenergy).
  - Average power demand: demand_kw = 4.0 * energy_15min_kwh.
  - Quality filter: pctmissing <= 10.0%.
  - Aggregate demand: Primary ground truth strictly requires all 19 apartments valid (553 intervals).
  - Sensitivity analysis (18/19 valid, 2,262 intervals) reported as partial sum; missing apartment never imputed.
  - Status: PASS.
Check 2.2: CEEW 15-Minute 5-Subinterval Aggregation
  - Aggregation window: S(T) = {T-12min, T-9min, T-6min, T-3min, T}.
  - Strict completeness rule: subinterval_count == 5, first == T-12min, last == T.
  - Any incomplete window assigned NaN (rejected: 0.31% Mathura, 0.13% Bareilly).
  - Average demand: demand_kw = 4.0 * sum(t_kwh).
  - Status: PASS.
Check 2.3: Operating State Disaggregation
  - Consumer zero load: voltage >= 150V, current == 0A, energy == 0 -> valid demand = 0.0 kW.
  - Grid outage: voltage < 50V, current == 0A, energy == 0 -> flagged with is_outage boolean.
  - Zero-load intervals preserved in dataset; outages disaggregated.
  - Status: PASS.
Check 2.4: Aggregate Cohort Eligibility Thresholds
  - Primary cohort analysis threshold: >= 90% fixed cohort reporting (Mathura: 9,813 intervals; Bareilly: 9,721 intervals).
  - Sensitivity cohort analysis threshold: 85% to <90% reporting (Mathura: 3,721 intervals; Bareilly: 10,706 intervals).
  - Sub-threshold intervals assigned NaN; partial sums never labeled as full measured grid demand.
  - Status: PASS.

========================================================================================================================
AUDIT 3 — WEATHER FEATURES
========================================================================================================================
Check 3.1: Provenance, Grid Centroids & Units
  - Source: Open-Meteo ERA5-Land Reanalysis API (CC BY 4.0, 0.1 deg grid resolution).
  - Grid Centroids: Delhi (28.5764, 77.2893), Mathura (27.5220, 77.7044), Bareilly (28.3656, 79.4779).
  - Offsets: Delhi 3.96 km, Mathura 4.46 km, Bareilly 4.65 km.
  - 9 variables acquired: 2m temperature (C), relative humidity (%), dew point (C), surface pressure (hPa),
    wind speed (m/s), direct radiation (W/m2), diffuse radiation (W/m2), cloud cover (%), precipitation (mm).
  - All 6 weather files match manifest SHA-256 hashes.
  - Status: PASS.
Check 3.2: Dual Weather Architecture Separation
  - Retrospective reanalysis stored under retro_* prefix (contemporaneous reanalysis).
  - Operational weather stored under operational_* prefix (enforcing 5-day publication latency).
  - Retrospective age tracked (0-60 min); operational age tracked (~120 hours).
  - Status: PASS.
Check 3.3: Causal Merge & Future Leakage Prevention
  - Merged using pd.merge_asof with direction="backward".
  - Operational cutoff: operational_cutoff = timestamp - 120h.
  - Mathematical assertion: max(t_weather) <= T. Zero future weather observations enter input features.
  - Status: PASS.

========================================================================================================================
AUDIT 4 — EDA SCIENTIFIC VALIDITY
========================================================================================================================
Check 4.1: Fig 3 Household Load Diversity Panel
  - Bareilly panel populated using genuinely observed data from meters BR02-BR06 (2019-08-28 to 2019-09-04).
  - All 5 meters exhibit 100% non-null valid readings (672/672 intervals).
  - No missing values converted to zero.
  - Pixel verification: 284,206 colored line pixels (15.19% panel density) confirmed in panel 3 bounding box.
  - Status: PASS.
Check 4.2: Sample Week Window Completeness
  - All 3 panels in Fig 3 contain exactly 672 timestamps (7 days x 96 intervals).
  - Window date spans explicitly embedded in subplot titles.
  - Status: PASS.
Check 4.3: Quadratic Temperature-Demand Regression Bounding
  - truncate=True enforced in sns.regplot; regression curve strictly bounded to observed domain [8.7, 43.8] C.
  - ax.set_ylim(bottom=0) enforced; prevented unconstrained parabola from extrapolating into negative demand.
  - Status: PASS.
Check 4.4: Correlation Statistics & Associational Labeling
  - Pearson r and Spearman rho computed on full valid pairwise observations and displayed in titles with sample size N.
    * Delhi (N=553): Temp r = +0.228, rho = +0.220 | Solar r = -0.217, rho = -0.300
    * Mathura (N=9,813): Temp r = +0.281, rho = +0.298 | Solar r = -0.303, rho = -0.420
    * Bareilly (N=9,721): Temp r = +0.498, rho = +0.514 | Solar r = -0.003, rho = +0.020
  - Regression curves explicitly labeled in plot legends as "Exploratory Association (2nd-Order Fit)" and
    "Exploratory Association (Linear Fit)" to prevent causal misinterpretation.
  - Status: PASS.
Check 4.5: Fig 5 Telemetry Reporting Labeling
  - X-axis and subplot titles renamed from "Active Cohort Ratio" to "Telemetry Reporting Ratio".
  - Status: PASS.

========================================================================================================================
AUDIT 5 — SEQUENCE GENERATION AND TEMPORAL BOUNDARIES
========================================================================================================================
Check 5.1: Sequence Dimensions & Slicing Continuity
  - Lookback L = 16 steps (4.0 hours) and forecast horizon H = 1 step (15 minutes).
  - Contiguous slicing requires all 17 steps [i, ..., i+16] valid; never crosses missing intervals or logging gaps.
  - Status: PASS.
Check 5.2: Meter-Wise Sequence Census Reconciliation
  - Meter-wise census exported to docs/meter_wise_sequence_census.csv across all 103 longitudinal series:
    * Mathura (38 meters): 1,536,402 total = 1,075,464 train + 608 purge1 + 230,441 val + 608 purge2 + 229,281 test. (Diff: 0)
    * Bareilly (46 meters): 2,668,842 total = 1,868,167 train + 736 purge1 + 400,306 val + 736 purge2 + 398,897 test. (Diff: 0)
    * Delhi (19 apts): 262,680 total = 183,866 train + 304 purge1 + 39,393 val + 304 purge2 + 38,813 test. (Diff: 0)
    * Aggregate Series: Mathura 9,005 (Diff: 0), Bareilly 8,585 (Diff: 0), Delhi 96 (Diff: 0).
  - Sequence yield sum across all individual meters: 4,467,924 contiguous sequences.
  - Reconciled discrepancy across all 106 audited series: 0.
  - Status: PASS.
Check 5.3: Boundary Purge Implementation & Non-Overlap
  - Discards 16 sequence samples across train-val and val-test boundaries.
  - Complete timestamp sets extracted and intersected across all 106 series (docs/all_series_boundary_purge_audit.csv).
  - Max cross-boundary timestamp overlap across all 106 series: 0 timestamps.
  - Boundary gap: 15.0 minutes (mathematically exact: t_val_first_in - t_train_last_target = t_{n+16} - t_{n+15} = 1 interval).
  - Status: PASS (106 / 106 series pass).

========================================================================================================================
AUDIT 6 — SCALING AND LEAKAGE PREVENTION
========================================================================================================================
Check 6.1: Chronological Splits & Scaler Fitting
  - Strict 70% Train, 15% Val, 15% Test chronological non-shuffled partition.
  - StandardScaler for features and targets fit strictly on training set.
  - Validation and test transformed using frozen training statistics.
  - Target scaler inverse recovery verified with 1e-5 precision.
  - Status: PASS.
Check 6.2: Temporal Leakage Audit
  - Target value y_{t+1} strictly excluded from input lookback X_{[t-15, ..., t]}.
  - Cyclical calendar encodings (hour_sin/cos, day_sin/cos) computed deterministically at origin T.
  - Identical chronological split rules applied per meter in individual panels.
  - Status: PASS.

========================================================================================================================
AUDIT 7 — BASELINES AND EVALUATION
========================================================================================================================
Check 7.1: Baseline Models Implementation
  - Naive Persistence: y_hat_{t+1} = y_t.
  - Seasonal Same-Time Previous-Day Persistence: y_hat_{t+1} = y_{t+1-96} (24-hour lag matching).
  - Status: PASS.
Check 7.2: Full-Test vs. Matched-Subset Benchmarking
  - Full-test set reports operational Naive baseline and Seasonal coverage ratio (46.06% Mathura, 41.61% Bareilly).
  - Matched-subset benchmark evaluates Naive and Seasonal persistence on identical test timestamps:
    * Mathura Aggregate (N=608): Naive RMSE = 1.2762 kW vs. Seasonal RMSE = 1.9059 kW (Skill Score: -1.2301)
    * Bareilly Aggregate (N=523): Naive RMSE = 2.3525 kW vs. Seasonal RMSE = 4.8224 kW (Skill Score: -3.2021)
  - Full results exported to artifacts/metrics/baseline_benchmarks.json.
  - Status: PASS.
Check 7.3: Power Engineering Metrics & Statistical Testing Guard
  - RMSE (kW), MAE (kW), Peak MAE (kW) in physical kW.
  - WAPE (%) and nRMSE (%) relative to mean load.
  - MAPE (%) zero-load guarded (< 0.01 kW).
  - Serial autocorrelation of overlapping sequence windows documented under ISS-008; hypothesis testing mandates
    Diebold-Mariano tests with lag correction or block bootstrap.
  - Status: PASS.

========================================================================================================================
AUDIT 8 — MODEL ARCHITECTURES
========================================================================================================================
Check 8.1: Architecture Implementation & Parameter Balance
  - 1D CNN (src/models/cnn1d.py): 3 Conv1D layers + BatchNorm + AdaptiveAvgPool1D + Dense regressor.
    Parameters: 52,225 (F=1) / 53,953 (F=10).
  - LSTM (src/models/lstm.py): 2-layer LSTM (hidden=64) + Last-step Dense regressor.
    Parameters: 52,545 (F=1) / 54,849 (F=10).
  - CNN-LSTM (src/models/cnn_lstm.py): Conv1D frontend + 2-layer LSTM + Dense regressor.
    Parameters: 69,057 (F=1) / 70,785 (F=10).
  - Balanced capacity (~52k-71k params) eliminates model size mismatch.
  - Tensor shapes: Input (B, 16, F), Output (B, 1). Single-step forecast (H=1).
  - Status: PASS.
Check 8.2: Train/Eval Mode & Seeds
  - Explicit model.train() and model.eval() modes with torch.no_grad() during evaluation.
  - Random seed 42 set across NumPy, PyTorch, and Python.
  - Status: PASS.

========================================================================================================================
AUDIT 9 — TRAINING PIPELINE
========================================================================================================================
Check 9.1: Optimization & Early Stopping
  - Loss function: HuberLoss(delta=1.0) (robust to electrical spike outliers).
  - Optimizer: AdamW(lr=1e-3, weight_decay=1e-4).
  - Scheduler: ReduceLROnPlateau(mode="min", factor=0.5, patience=4-5, min_lr=1e-6).
  - Gradient clipping: max_norm=1.0.
  - Early stopping: patience 8-10 epochs based strictly on validation loss.
  - Best checkpoint restoration: model.load_state_dict(best_weights).
  - Test set evaluated strictly after final model restoration.
  - Status: PASS.
Check 9.2: Colab Environment Readiness & Resilient Paths
  - GPU acceleration: device = torch.device("cuda" if torch.cuda.is_available() else "cpu").
  - Google Colab detection and optional Drive mount instructions embedded in notebook cell 2.
  - Robust path resolution: checks both local and Colab working directories.
  - Status: PASS.

========================================================================================================================
AUDIT 10 — REPRODUCIBILITY AND DOCUMENTATION
========================================================================================================================
Check 10.1: Package Dependencies & Configuration Sync
  - requirements.txt contains all core dependencies (requests, numpy, pandas, scipy, scikit-learn, pyarrow, pyyaml, matplotlib, seaborn, streamlit, pytest).
  - config/config.yaml synchronized with reconciled sequence census, 4-hour lookback scope, and dual weather feature sets.
  - Status: PASS.
Check 10.2: Known Issues Backlog
  - docs/KNOWN_ISSUES.md tracks ISS-001 through ISS-009.
  - Zero uncharacterized blockers remain.
  - Status: PASS.
```

---

## Final Pre-Training Decision

```
========================================================================================================================
FINAL AUDIT DECISION: GO
========================================================================================================================
All 31 audit checklist items across Audits 1 through 10 have PASSED.
Zero critical blockers identified.
All raw datasets under data/raw/ remain 100% immutable and verified against SHA-256 manifests.
All target construction, sequence slicing, boundary purge, leakage prevention, baseline benchmarks,
and model architectures are certified scientifically valid and mathematically sound.

AUTHORIZATION: Deep learning model training across the full experimental matrix is authorized to proceed
on the Google Colab GPU runtime.
========================================================================================================================
```

