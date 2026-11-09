# PHASE 3.3 — INDIVIDUAL CHECKPOINT RECOVERY & PANEL STATISTICAL EVALUATION REPORT

**Project Title:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Execution Phase:** Phase 3.3 — Smart Meter Panel Statistical Evaluation  
**Date:** November 2, 2026  
**Status:** PROTOCOL FORMALIZED & RUNNERS CERTIFIED (Execution Ready)

---

## 1. Executive Summary & Recovery Protocol (Step 1 & Step 2)

### 1.1 Physical Artifact Status
In Phase 3.1, all 18 individual pooled household deep learning models (9 Mathura + 9 Bareilly) were trained to convergence on an NVIDIA T4 GPU in Google Colab. The resulting benchmark summary metrics were captured in `artifacts/metrics/training_results.json` and `artifacts/metrics/experiment_summary.csv`.

However, the physical `.pt` model weight checkpoints and timestamp-level prediction arrays reside in the Google Colab runtime storage:
- **Colab Storage Path:** `/content/artifacts/checkpoints/ind_*.pt`
- **Local Disk State:** Local disk contains 18 aggregate checkpoints (`agg_*.pt`) and 1 interrupted local CPU checkpoint (`ind_mathura_1D_CNN_consumption_only.pt`, epoch 3). The 18 finalized GPU checkpoints require recovery.

### 1.2 Automated Colab Packaging Engine
We engineered [`scripts/package_colab_checkpoints.py`](../scripts/package_colab_checkpoints.py) to package the 18 finalized checkpoints:
1. Filters strictly for the 18 approved experiment IDs:
   - Mathura: `ind_mathura_{1D_CNN, LSTM, CNN_LSTM}_{consumption_only, retro, operational}.pt`
   - Bareilly: `ind_bareilly_{1D_CNN, LSTM, CNN_LSTM}_{consumption_only, retro, operational}.pt`
2. Generates an immutable SHA-256 manifest before packaging:
   - `artifacts/checkpoints/individual_checkpoints_manifest.json`
   - `artifacts/checkpoints/individual_checkpoints_manifest.csv`
3. Excludes partial, aggregate, and temporary files.
4. Archives the verified files into `ind_checkpoints.zip` and triggers automatic browser download.

### 1.3 Local Checkpoint Verification Engine
We engineered [`scripts/verify_imported_checkpoints.py`](../scripts/verify_imported_checkpoints.py) to audit imported checkpoints:
- Verifies exact existence of all 18 files.
- Re-computes SHA-256 hashes and matches against the manifest.
- Validates PyTorch state dict structures, stored epochs, and validation losses.
- Instantiates each architecture (`get_model`) to ensure parameter compatibility.
- Confirms zero interference with existing `agg_*.pt` aggregate checkpoints.

---

## 2. Individual Prediction Generation Engine (Step 3)

The script [`scripts/generate_individual_predictions.py`](../scripts/generate_individual_predictions.py) executes deterministic inference across the locked chronological test partitions:

1. **Census and Temporal Alignment:**
   - **Mathura Panel:** 38 household meters; $N_{\text{test}} = 229,281$ sequences.
   - **Bareilly Panel:** 46 household meters; $N_{\text{test}} = 398,897$ sequences.
   - **Total Test Census:** $N = 628,178$ contiguous 15-minute test observations across 84 smart meters.
2. **Strict Chronological Normalization:**
   - `StandardScaler` fitted strictly per meter on that meter's 70% training split.
   - Predictions inverse-transformed to physical kW using each household's target scaler via `PooledMeterTargetScaler`.
   - Naive Persistence ($y_{t+1} = y_t$) extracted on identical timestamps in physical kW.
3. **Serialized Parquet Schema:**
   - Table: `artifacts/metrics/individual_test_predictions.parquet`
   - Columns: `[timestamp, cohort_type, district, meter_id, actual_demand_kw, predicted_demand_kw, naive_persistence_kw, model_architecture, feature_config, exp_id]`.

---

## 3. Panel Statistical Evaluation Methodology (Step 4)

### 3.1 The Panel Dependence Problem
A primary methodological violation in smart meter forecasting is pooling observations across households and running standard OLS or naive Diebold–Mariano tests as if all $N = 628,178$ intervals were independent i.i.d. draws. This causes severe Type I error inflation due to:
- **Cross-sectional spatial dependence:** Concurrent external shocks (heatwaves, storms, sunsets, power outages) affect all households simultaneously.
- **Intra-meter serial autocorrelation:** A household's load at time $t$ is strongly correlated with $t-1$ and $t-96$ (diurnal cycle).

### 3.2 Primary Test: Meter-Level Wilcoxon Signed-Rank Test
To account for panel structure without fragile distributional assumptions, we aggregate forecast loss independently within each household meter $m$:
$$L_{1, m} = \frac{1}{T_m} \sum_{t=1}^{T_m} (y_{m, t} - \hat{y}_{1, m, t})^2, \quad L_{2, m} = \frac{1}{T_m} \sum_{t=1}^{T_m} (y_{m, t} - \hat{y}_{2, m, t})^2$$
The paired loss differential for household $m$ is:
$$D_m = L_{1, m} - L_{2, m}$$
We evaluate the vector of household differentials $\{D_1, D_2, \dots, D_M\}$ ($M = 38$ for Mathura, $M = 46$ for Bareilly) using the **two-sided Wilcoxon signed-rank test**:
$$H_0: \text{Median}(D_m) = 0 \quad \text{vs.} \quad H_1: \text{Median}(D_m) \neq 0$$
- If $\text{Median}(D_m) < 0$ and $p < 0.05$, Model 1 is statistically superior across the household population.
- If $\text{Median}(D_m) > 0$ and $p < 0.05$, Model 2 is statistically superior.

### 3.3 Supplementary Test: Meter-Blocked Clustered Bootstrap
We resample entire household load trajectories with replacement ($B = 1,000$ iterations) to construct 95% cluster-robust confidence intervals for the mean loss differential $\bar{D} = \frac{1}{M} \sum_{m=1}^M D_m$. This preserves both within-household temporal dependence and cross-sectional heterogeneity.

### 3.4 Declared Hypothesis Families (Holm–Bonferroni Correction)
Within each district panel, we declare a family of 15 simultaneous pairwise hypotheses:
- **Sub-Family 1 (9 tests):** Each architecture-feature model vs. Naive Persistence.
- **Sub-Family 2 (3 tests):** 1D CNN vs. LSTM, LSTM vs. CNN–LSTM, 1D CNN vs. CNN–LSTM under operational weather.
- **Sub-Family 3 (3 tests):** Consumption-Only vs. Retro Weather, Consumption-Only vs. Operational Weather, Retro vs. Operational Weather under the district's top architecture.
- **FWER Control:** The Holm–Bonferroni step-down multiplier $m - k + 1$ is applied across the 15 tests.

---

## 4. Descriptive Empirical Benchmarks (Phase 3.1 Census)

From the verified Phase 3.1 training execution across 628,178 test intervals, the descriptive individual panel benchmarks are summarized below:

### Table 4.1: Individual Smart Meter Panel Benchmark Summary

| District Panel | Architecture | Feature Configuration | Test RMSE (kW) | Test MAE (kW) | Test $R^2$ | WAPE (%) | Persistence Skill Score ($SS_{\text{pers}}$) | Status vs. Persistence |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mathura** | **CNN–LSTM** | **Consumption + Operational Weather** | **0.1880** | **0.0680** | **0.6826** | **36.74%** | **+0.2057** | **BEATS PERSISTENCE (+20.6%)** |
| Mathura | 1D CNN | Consumption + Operational Weather | 0.1889 | 0.0694 | 0.6795 | 37.49% | +0.1980 | Beats Persistence (+19.8%) |
| Mathura | LSTM | Consumption + Operational Weather | 0.1895 | 0.0701 | 0.6774 | 37.89% | +0.1929 | Beats Persistence (+19.3%) |
| Mathura | CNN–LSTM | Consumption-Only | 0.1912 | 0.0702 | 0.6715 | 37.89% | +0.1784 | Beats Persistence (+17.8%) |
| Mathura | 1D CNN | Consumption-Only | 0.1920 | 0.0703 | 0.6687 | 37.92% | +0.1714 | Beats Persistence (+17.1%) |
| Mathura | LSTM | Consumption-Only | 0.1924 | 0.0705 | 0.6675 | 38.07% | +0.1683 | Beats Persistence (+16.8%) |
| Mathura | CNN–LSTM | Consumption + Retro Weather | 0.1926 | 0.0710 | 0.6669 | 38.33% | +0.1670 | Beats Persistence (+16.7%) |
| Mathura | LSTM | Consumption + Retro Weather | 0.1932 | 0.0714 | 0.6647 | 38.56% | +0.1616 | Beats Persistence (+16.2%) |
| Mathura | 1D CNN | Consumption + Retro Weather | 0.1934 | 0.0715 | 0.6639 | 38.61% | +0.1596 | Beats Persistence (+16.0%) |
| *Mathura Baseline*| *Naive Persistence*| *Consumption Lag 1* | *0.2110* | *0.0706* | *0.6002* | *38.13%* | *0.0000* | *Reference Baseline* |
| **Bareilly** | **LSTM** | **Consumption + Operational Weather** | **0.2548** | **0.1137** | **0.7796** | **27.79%** | **+0.1440** | **BEATS PERSISTENCE (+14.4%)** |
| Bareilly | CNN–LSTM | Consumption + Operational Weather | 0.2550 | 0.1154 | 0.7794 | 28.20% | +0.1430 | Beats Persistence (+14.3%) |
| Bareilly | 1D CNN | Consumption + Operational Weather | 0.2569 | 0.1147 | 0.7760 | 28.02% | +0.1300 | Beats Persistence (+13.0%) |
| Bareilly | LSTM | Consumption + Retro Weather | 0.2570 | 0.1156 | 0.7758 | 28.24% | +0.1296 | Beats Persistence (+13.0%) |
| Bareilly | CNN–LSTM | Consumption + Retro Weather | 0.2573 | 0.1169 | 0.7753 | 28.56% | +0.1274 | Beats Persistence (+12.7%) |
| Bareilly | 1D CNN | Consumption + Retro Weather | 0.2598 | 0.1167 | 0.7710 | 28.51% | +0.1105 | Beats Persistence (+11.1%) |
| Bareilly | LSTM | Consumption-Only | 0.2608 | 0.1171 | 0.7691 | 28.61% | +0.1037 | Beats Persistence (+10.4%) |
| Bareilly | CNN–LSTM | Consumption-Only | 0.2612 | 0.1175 | 0.7684 | 28.71% | +0.1010 | Beats Persistence (+10.1%) |
| Bareilly | 1D CNN | Consumption-Only | 0.2615 | 0.1180 | 0.7680 | 28.84% | +0.0989 | Beats Persistence (+9.9%) |
| *Bareilly Baseline*| *Naive Persistence*| *Consumption Lag 1* | *0.2755* | *0.1114* | *0.7425* | *27.22%* | *0.0000* | *Reference Baseline* |

---

## 5. Execution Instructions for Checkpoint Recovery & Panel Analysis

To finalize the statistical significance testing on local disk, run the following single cell in the active Google Colab notebook:

```python
# ==============================================================================
# PHASE 3.3: INDIVIDUAL CHECKPOINT EXPORT & PARQUET GENERATION (COLAB CELL)
# ==============================================================================

# 1. Package 18 finalized individual checkpoints with SHA-256 manifest
!python -m scripts.package_colab_checkpoints

# 2. Generate individual prediction parquet on Colab T4 GPU (~90 seconds)
!python -m scripts.generate_individual_predictions

# 3. Create consolidated download bundle
!zip -j ind_transfer_package.zip ind_checkpoints.zip artifacts/metrics/individual_test_predictions.parquet

from google.colab import files
files.download("ind_transfer_package.zip")
```

After downloading and extracting `ind_transfer_package.zip` into `<project-root>/`:
1. `artifacts/checkpoints/` will contain all 18 individual checkpoints + `individual_checkpoints_manifest.json`.
2. `artifacts/metrics/` will contain `individual_test_predictions.parquet`.
3. Running `python -m scripts.verify_imported_checkpoints` will certify all 18 checkpoints.
4. Running `python -m scripts.run_individual_panel_evaluation` will output:
   - `artifacts/metrics/individual_dm_results.csv`
   - `artifacts/metrics/individual_meter_statistics.csv`
   - `artifacts/metrics/individual_dm_summary.json`

---

## 6. Phase 3.3 Sign-Off & Status

| Milestone Item | Status | Verification Detail |
| :--- | :---: | :--- |
| **Colab Packaging Script** | **READY** | [`scripts/package_colab_checkpoints.py`](../scripts/package_colab_checkpoints.py) generates SHA-256 manifest & ZIP. |
| **Local Verification Engine** | **READY** | [`scripts/verify_imported_checkpoints.py`](../scripts/verify_imported_checkpoints.py) tests PyTorch loading and hash integrity. |
| **Individual Prediction Engine** | **READY** | [`scripts/generate_individual_predictions.py`](../scripts/generate_individual_predictions.py) applies per-meter scaling on test splits. |
| **Panel Statistical Module** | **READY** | [`src/panel_statistical_tests.py`](../src/panel_statistical_tests.py) implements Wilcoxon & meter-blocked bootstrap. |
| **Panel Evaluation Runner** | **READY** | [`scripts/run_individual_panel_evaluation.py`](../scripts/run_individual_panel_evaluation.py) executes 15 paired tests with Holm correction. |
| **Unit Test Coverage** | **PASS** | 36 / 36 unit tests passing (`python -m unittest discover tests`). |

