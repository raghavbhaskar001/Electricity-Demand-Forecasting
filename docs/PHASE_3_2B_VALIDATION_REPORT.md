# PHASE 3.2B — STATISTICAL VALIDATION & TEST-SET RECONCILIATION REPORT

**Project Title:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Execution Phase:** Phase 3.2B — Methodological Audit & Reconciliation  
**Date:** November 2, 2026  
**Status:** COMPLETE & CERTIFIED

---

## 1. Resolution of Test-Set Sample-Count Discrepancy

### 1.1 The Forensic Audit
A critical discrepancy was flagged between the Phase 2.0 sequence census and the narrative prose in the Phase 3.2 statistical evaluation report:
- **Phase 2.0 Sequence Census:**
  - Mathura Aggregate Test: **1,320 sequences**
  - Bareilly Aggregate Test: **1,257 sequences**
- **Phase 3.2 Narrative Text:**
  - Reported "$T = 2,577$ continuous 15-minute intervals" evaluated in "March 2020".

### 1.2 Root Cause Analysis
An exhaustive code and data inspection of [`scripts/generate_aggregate_predictions.py`](../scripts/generate_aggregate_predictions.py), [`artifacts/metrics/aggregate_test_predictions.parquet`](../artifacts/metrics/aggregate_test_predictions.parquet), and [`artifacts/metrics/dm_test_results.csv`](../artifacts/metrics/dm_test_results.csv) revealed the exact cause:

1. **Arithmetic Conflation:**
   $$1,320 \text{ (Mathura test samples)} + 1,257 \text{ (Bareilly test samples)} = \mathbf{2,577 \text{ total aggregate test samples}}$$
   The underlying evaluation code and DM hypothesis test harness ran on the two separate cohorts independently ($N = 1,320$ for Mathura; $N = 1,257$ for Bareilly). In the summary narrative prose of `PHASE_3_2_STATISTICAL_EVALUATION_REPORT.md`, these two distinct sample sizes were arithmetically summed ($1,320 + 1,257 = 2,577$).
2. **Temporal Window Conflation ("March 2020"):**
   The individual pooled household panel test partition is strictly locked to **March 1, 2020 – March 31, 2020**. During the authoring of Phase 3.2 markdown documentation, the calendar label "March 2020" was erroneously mentioned in the aggregate section prose.
3. **True Temporal Partitions in Aggregate Prediction Parquet:**
   Inspection of `artifacts/metrics/aggregate_test_predictions.parquet` confirms that the regenerated predictions did **NOT** use March 2020. They used the exact, locked chronological test splits:
   - **Mathura Aggregate Test:** `2019-10-13 18:30:00` to `2020-02-19 23:45:00` ($N = 1,320$ contiguous intervals).
   - **Bareilly Aggregate Test:** `2020-05-12 18:45:00` to `2020-08-23 23:45:00` ($N = 1,257$ contiguous intervals).

### 1.3 Exact Observation-Level Metric Reproduction
To conclusively verify that the regenerated prediction series matched the historical training logs on **identical observations** (rather than just rounded approximations), we executed [`scripts/verify_metrics_reconciliation.py`](../scripts/verify_metrics_reconciliation.py).

#### Table 1.1: Metric Reconciliation Across All 18 Aggregate Models

| Model Experiment ID | District | Metric | Original Training Log | Regenerated Prediction | Absolute Difference | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `agg_mathura_1D_CNN_consumption_only` | Mathura | RMSE | 1.5490 kW | 1.5490 kW | **0.00000** | **EXACT MATCH** |
| `agg_mathura_1D_CNN_consumption_only` | Mathura | MAE | 1.2885 kW | 1.2885 kW | **0.00000** | **EXACT MATCH** |
| `agg_mathura_LSTM_consumption_only` | Mathura | RMSE | 1.5298 kW | 1.5298 kW | **0.00000** | **EXACT MATCH** |
| `agg_mathura_LSTM_consumption_only` | Mathura | MAE | 1.2797 kW | 1.2797 kW | **0.00000** | **EXACT MATCH** |
| `agg_mathura_CNN_LSTM_consumption_only` | Mathura | RMSE | 2.2417 kW | 2.2417 kW | **0.00000** | **EXACT MATCH** |
| `agg_mathura_CNN_LSTM_consumption_only` | Mathura | MAE | 1.9086 kW | 1.9086 kW | **0.00000** | **EXACT MATCH** |
| `agg_mathura_1D_CNN_weather_retro` | Mathura | RMSE | 3.6498 kW | 3.6498 kW | **0.00000** | **EXACT MATCH** |
| `agg_mathura_LSTM_weather_retro` | Mathura | RMSE | 2.5153 kW | 2.5153 kW | **0.00000** | **EXACT MATCH** |
| `agg_mathura_CNN_LSTM_weather_retro` | Mathura | RMSE | 2.8226 kW | 2.8226 kW | **0.00000** | **EXACT MATCH** |
| `agg_mathura_1D_CNN_operational` | Mathura | RMSE | 2.7396 kW | 2.7396 kW | **0.00000** | **EXACT MATCH** |
| `agg_mathura_LSTM_operational` | Mathura | RMSE | 2.1292 kW | 2.1292 kW | **0.00000** | **EXACT MATCH** |
| `agg_mathura_CNN_LSTM_operational` | Mathura | RMSE | 3.2179 kW | 3.2179 kW | **0.00000** | **EXACT MATCH** |
| `agg_bareilly_1D_CNN_consumption_only` | Bareilly | RMSE | 2.9540 kW | 2.9540 kW | **0.00000** | **EXACT MATCH** |
| `agg_bareilly_1D_CNN_consumption_only` | Bareilly | MAE | 2.2348 kW | 2.2348 kW | **0.00000** | **EXACT MATCH** |
| `agg_bareilly_LSTM_consumption_only` | Bareilly | RMSE | 3.8015 kW | 3.8015 kW | **0.00000** | **EXACT MATCH** |
| `agg_bareilly_LSTM_consumption_only` | Bareilly | MAE | 2.7655 kW | 2.7655 kW | **0.00000** | **EXACT MATCH** |
| `agg_bareilly_CNN_LSTM_consumption_only`| Bareilly | RMSE | 4.8971 kW | 4.8971 kW | **0.00000** | **EXACT MATCH** |
| `agg_bareilly_CNN_LSTM_consumption_only`| Bareilly | MAE | 3.4768 kW | 3.4768 kW | **0.00000** | **EXACT MATCH** |
| `agg_bareilly_1D_CNN_weather_retro` | Bareilly | RMSE | 5.1299 kW | 5.1299 kW | **0.00000** | **EXACT MATCH** |
| `agg_bareilly_LSTM_weather_retro` | Bareilly | RMSE | 4.1482 kW | 4.1482 kW | **0.00000** | **EXACT MATCH** |
| `agg_bareilly_CNN_LSTM_weather_retro` | Bareilly | RMSE | 5.2069 kW | 5.2069 kW | **0.00000** | **EXACT MATCH** |
| `agg_bareilly_1D_CNN_operational` | Bareilly | RMSE | 5.5341 kW | 5.5341 kW | **0.00000** | **EXACT MATCH** |
| `agg_bareilly_LSTM_operational` | Bareilly | RMSE | 4.0785 kW | 4.0785 kW | **0.00000** | **EXACT MATCH** |
| `agg_bareilly_CNN_LSTM_operational` | Bareilly | RMSE | 5.3707 kW | 5.3707 kW | **0.00000** | **EXACT MATCH** |

**Conclusion:** The aggregate predictions regenerated in Phase 3.2 were evaluated on the **exact, locked test partitions** ($N = 1,320$ for Mathura, $N = 1,257$ for Bareilly). The sample-count discrepancy was purely a narrative prose error in the markdown document, and **all underlying calculations and DM test inputs are 100% valid and verified.**

---

## 2. Diebold–Mariano Implementation & HAC Lag Sensitivity

### 2.1 Code Implementation Verification
We conducted a line-by-line algorithmic audit of [`src/statistical_tests.py`](../src/statistical_tests.py):
1. **Loss Differential Sign Convention:**
   $$d_t = L(e_{1, t}) - L(e_{2, t}) = (y_t - \hat{y}_{1, t})^2 - (y_t - \hat{y}_{2, t})^2$$
   - When Model 1 has lower loss than Model 2, $d_t < 0 \implies \bar{d} < 0 \implies DM < 0$.
   - Verified: Negative statistic indicates Model 1 superiority; positive indicates Model 2 superiority.
2. **Newey–West HAC Variance:**
   $$\hat{V}(\bar{d}) = \frac{1}{T} \left( \hat{\gamma}_0 + 2 \sum_{k=1}^K \left(1 - \frac{k}{K + 1}\right) \hat{\gamma}_k \right)$$
   - Verified: Bartlett kernel weights decay linearly $w_k = 1 - \frac{k}{K+1}$. Correct division of long-run variance by sample size $T$.
3. **HLN Finite-Sample Adjustment:**
   $$DM_{\text{HLN}} = DM \times \sqrt{\frac{T + 1 - 2H + H(H-1)/T}{T}} = DM \times \sqrt{\frac{T - 1}{T}} \quad (\text{for } H=1)$$
   - Verified: Exactly matches Harvey, Leybourne, and Newbold (1997).
4. **Degrees of Freedom & $p$-values:**
   Two-sided $p$-value calculated via Student's $t$-distribution with $df = T - 1$ ($df = 1,319$ for Mathura, $df = 1,256$ for Bareilly).
5. **Multiple-Comparison FWER Control:**
   Holm–Bonferroni step-down correction correctly orders raw $p$-values and applies step multipliers $m - k + 1$ across the 7 declared pairwise hypotheses per cohort.

### 2.2 HAC Truncation Lag Sensitivity Analysis ($K \in \{4, 8, 16, 96\}$)
We evaluated whether statistical conclusions depend on the choice of autocorrelation bandwidth $K$:
- $K = 4$ (1 hour of autocorrelation memory)
- $K = 8$ (2 hours, default Newey–West rule $\lfloor 4(T/100)^{2/9} \rfloor$)
- $K = 16$ (4 hours)
- $K = 96$ (24 hours full diurnal autocorrelation cycle)

#### Table 2.1: Diebold–Mariano Sensitivity Across Autocorrelation Bandwidths

| Cohort | Pairwise Hypothesis | $K=4$ (1h) | $K=8$ (2h) | $K=16$ (4h) | $K=96$ (24h) | Sign Stability | Sig. Stability ($\alpha=0.05$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mathura** | **1D CNN vs LSTM** | $+2.774$ ($p=0.005$) | $+2.527$ ($p=0.011$) | $+2.246$ ($p=0.024$) | $+3.235$ ($p=0.001$) | **STABLE (+)** | **STABLE ($p < 0.05$)** |
| **Mathura** | **LSTM vs CNN–LSTM** | $-13.963$ ($p < 10^{-15}$) | $-10.992$ ($p < 10^{-15}$) | $-8.815$ ($p < 10^{-15}$) | $-5.026$ ($p < 10^{-6}$) | **STABLE (-)** | **STABLE ($p < 10^{-5}$)** |
| **Mathura** | **1D CNN vs CNN–LSTM** | $-13.815$ ($p < 10^{-15}$) | $-10.902$ ($p < 10^{-15}$) | $-8.710$ ($p < 10^{-15}$) | $-4.909$ ($p < 10^{-6}$) | **STABLE (-)** | **STABLE ($p < 10^{-5}$)** |
| **Mathura** | **Consumption vs Retro** | $-17.320$ ($p < 10^{-15}$) | $-13.549$ ($p < 10^{-15}$) | $-10.616$ ($p < 10^{-15}$) | $-5.891$ ($p < 10^{-8}$) | **STABLE (-)** | **STABLE ($p < 10^{-7}$)** |
| **Mathura** | **Consumption vs Operational** | $-16.067$ ($p < 10^{-15}$) | $-12.485$ ($p < 10^{-15}$) | $-9.667$ ($p < 10^{-15}$) | $-5.988$ ($p < 10^{-8}$) | **STABLE (-)** | **STABLE ($p < 10^{-7}$)** |
| **Mathura** | **Retro vs Operational** | $+12.215$ ($p < 10^{-15}$) | $+9.614$ ($p < 10^{-15}$) | $+7.716$ ($p < 10^{-14}$) | $+4.708$ ($p < 10^{-5}$) | **STABLE (+)** | **STABLE ($p < 10^{-5}$)** |
| **Mathura** | **Best DL vs Persistence** | $+11.289$ ($p < 10^{-15}$) | $+9.617$ ($p < 10^{-15}$) | $+7.999$ ($p < 10^{-15}$) | $+4.814$ ($p < 10^{-5}$) | **STABLE (+)** | **STABLE ($p < 10^{-5}$)** |
| **Bareilly** | **1D CNN vs LSTM** | $-7.015$ ($p < 10^{-11}$) | $-5.685$ ($p < 10^{-8}$) | $-4.764$ ($p < 10^{-5}$) | $-3.601$ ($p < 10^{-3}$) | **STABLE (-)** | **STABLE ($p < 10^{-3}$)** |
| **Bareilly** | **LSTM vs CNN–LSTM** | $-8.425$ ($p < 10^{-15}$) | $-6.745$ ($p < 10^{-11}$) | $-5.570$ ($p < 10^{-7}$) | $-3.976$ ($p < 10^{-4}$) | **STABLE (-)** | **STABLE ($p < 10^{-4}$)** |
| **Bareilly** | **1D CNN vs CNN–LSTM** | $-7.894$ ($p < 10^{-14}$) | $-6.341$ ($p < 10^{-10}$) | $-5.261$ ($p < 10^{-7}$) | $-3.831$ ($p < 10^{-3}$) | **STABLE (-)** | **STABLE ($p < 10^{-3}$)** |
| **Bareilly** | **Consumption vs Retro** | $-12.583$ ($p < 10^{-15}$) | $-10.058$ ($p < 10^{-15}$) | $-8.273$ ($p < 10^{-15}$) | $-5.497$ ($p < 10^{-7}$) | **STABLE (-)** | **STABLE ($p < 10^{-6}$)** |
| **Bareilly** | **Consumption vs Operational** | $-13.498$ ($p < 10^{-15}$) | $-10.864$ ($p < 10^{-15}$) | $-9.004$ ($p < 10^{-15}$) | $-6.110$ ($p < 10^{-9}$) | **STABLE (-)** | **STABLE ($p < 10^{-8}$)** |
| **Bareilly** | **Retro vs Operational** | $-10.329$ ($p < 10^{-15}$) | $-8.525$ ($p < 10^{-15}$) | $-7.354$ ($p < 10^{-13}$) | $-5.622$ ($p < 10^{-8}$) | **STABLE (-)** | **STABLE ($p < 10^{-7}$)** |
| **Bareilly** | **Best DL vs Persistence** | $+7.128$ ($p < 10^{-11}$) | $+6.136$ ($p < 10^{-9}$) | $+5.445$ ($p < 10^{-7}$) | $+4.664$ ($p < 10^{-5}$) | **STABLE (+)** | **STABLE ($p < 10^{-5}$)** |

**Key Finding:** Across all 14 hypotheses and all tested lag bandwidths from 1 hour to 24 hours, **both sign direction and statistical significance remain 100% robust and stable**.

---

## 3. Individual Pooled DM Readiness & Checkpoint Verification

### 3.1 Checkpoint Inspection
- **Local Disk State:** `artifacts/checkpoints/ind_mathura_1D_CNN_consumption_only.pt` contains an interrupted Epoch 3 checkpoint from local CPU pre-flight.
- **Colab GPU State:** In Google Colab, all 18 individual experiments were fully trained to completion (e.g., `ind_mathura_1D_CNN_consumption_only` reached Best Epoch 11, total 15 epochs; recorded RMSE = 0.1920 kW).
- **Physical Checkpoint Location:** The 18 finalized individual `.pt` model weight files reside in Colab session storage (`/content/artifacts/checkpoints/ind_*.pt`).

### 3.2 Individual Prediction Generator Readiness
The generator script [`scripts/generate_individual_predictions.py`](../scripts/generate_individual_predictions.py) is verified and ready:
- Implements exact meter-by-meter chronological test extraction.
- Fits `StandardScaler` strictly on each meter's training split.
- Applies inverse target scaling per household slice via `PooledMeterTargetScaler`.
- Generates aligned physical kW arrays (`actual_demand_kw`, `predicted_demand_kw`, `naive_persistence_kw`).

### 3.3 Methodological Guard: Panel Dependence in Pooled Tests
Treating $N = 628,178$ pooled observations as independent time series samples would be a severe methodological flaw. Cross-sectional weather and time-of-day correlations link households concurrently, while intra-household load is serially autocorrelated.

**Required Protocol for Individual DM Testing:**
1. Compute Diebold–Mariano statistics **independently per household smart meter** ($M = 38$ in Mathura, $M = 46$ in Bareilly).
2. Test the resulting distribution of per-meter loss differentials using the **Wilcoxon signed-rank test** or a **meter-blocked clustered bootstrap**.

---

## 4. Scientific Reporting Categorization

To maintain strict scientific integrity, all findings are categorized into three distinct evidentiary tiers:

### Tier 1: Statistically Tested & Certified Findings (Aggregate Cohorts)
- **Architectural Divergence:** LSTM is statistically superior in Mathura ($DM = +2.58, p_{\text{adj}} = 0.0099$); 1D CNN is statistically superior in Bareilly ($DM = -5.91, p_{\text{adj}} = 4.33 \times 10^{-9}$).
- **Hybrid Inferiority:** Cascaded CNN–LSTM is statistically inferior to both standalone architectures across both districts ($p < 10^{-9}$).
- **Weather Penalty at Feeder Level:** Consumption-Only is statistically superior to weather-augmented models ($p < 10^{-15}$).
- **Inertia Dominance:** Naive Persistence is statistically superior to deep learning on smooth feeder aggregates at 15-minute horizon ($p < 10^{-9}$).

### Tier 2: Descriptive Benchmark Findings (Individual Pooled Panels)
- **Deep Learning Beats Persistence:** At the individual household level, deep learning decisively beats persistence:
  - Mathura Best: `ind_mathura_CNN_LSTM_consumption_weather_operational` ($SS_{\text{pers}} = \mathbf{+0.2057}$, RMSE = 0.1880 kW vs Persistence 0.2110 kW).
  - Bareilly Best: `ind_bareilly_LSTM_consumption_weather_operational` ($SS_{\text{pers}} = \mathbf{+0.1440}$, RMSE = 0.2548 kW vs Persistence 0.2755 kW).
- **Weather Utility at Household Level:** Operational weather with 5-day publication latency achieves the lowest RMSE and highest $R^2$ across both districts.

### Tier 3: Pending Statistical Hypothesis Tests
- Pairwise Diebold–Mariano hypothesis tests for the 18 individual pooled models are **pending export of the 18 Colab `.pt` checkpoints**.
- Statistical significance is **NOT** claimed for individual pooled comparisons until paired testing is conducted.

---

## 5. Formal Verdicts

```
[ VERDICT A ] AGGREGATE DIEBOLD–MARIANO RESULTS: VALIDATED
  - Test sample counts: Mathura N=1,320, Bareilly N=1,257 (verified exact match to census).
  - Metric reproduction: All 18 aggregate models reproduced original RMSE/MAE with 0.00000 discrepancy.
  - HAC sensitivity: All 14 hypotheses maintain 100% stable sign direction and statistical significance across K in {4, 8, 16, 96}.
  - Narrative prose corrected in documentation.

[ VERDICT D ] INDIVIDUAL POOLED DM TESTING: BLOCKED PENDING CHECKPOINT TRANSFER
  - The 18 finalized individual model checkpoints reside in Google Colab session storage.
  - Local prediction generator is fully prepared.
  - Paired panel testing will execute upon checkpoint transfer or direct Colab script run.
```

