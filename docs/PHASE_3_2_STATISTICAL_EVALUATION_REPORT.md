# PHASE 3.2 — STATISTICAL SIGNIFICANCE & DIEBOLD–MARIANO EVALUATION REPORT

**Project Title:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Evaluation Phase:** Phase 3.2 — Rigorous Statistical Hypothesis Testing  
**Date:** November 2, 2026  
**Status:** COMPLETE (Aggregate Hypotheses Certified; Individual Pooled Protocol Formalized)

---

## 1. Executive Summary & Prediction Artifact Audit

In Phase 3.2, we transitioned from deterministic point metric reporting (RMSE, MAE, $R^2$) to formal statistical hypothesis testing. We evaluated whether performance differentials among the deep learning models (1D CNN, LSTM, CNN–LSTM), input feature representations (Consumption-Only, Retrospective Weather, Operational Weather), and baseline persistence are statistically significant rather than stochastic artifacts of test sample noise.

### 1.1 Prediction Artifact Audit (Step 1)
- **Aggregate Cohorts (Mathura & Bareilly):**
  - **Audit Status:** Certified and fully regenerated on local disk without retraining.
  - **Checkpoints Verified:** All 18 aggregate `.pt` checkpoint files in `artifacts/checkpoints/agg_*.pt` were loaded into evaluation mode (`model.eval()`).
  - **Prediction Series:** Deterministic inference was conducted over the locked chronological held-out test splits:
    - **Mathura Aggregate Test:** 2019-10-13 18:30:00 to 2020-02-19 23:45:00 ($T = 1,320$ continuous 15-minute intervals).
    - **Bareilly Aggregate Test:** 2020-05-12 18:45:00 to 2020-08-23 23:45:00 ($T = 1,257$ continuous 15-minute intervals).
  - **Verification:** Regenerated test RMSE and MAE matched historical training logs down to $0.00000$ precision on identical test samples (e.g., `agg_mathura_1D_CNN_consumption_only` RMSE = 1.5490 kW; `agg_mathura_LSTM_consumption_only` RMSE = 1.5298 kW).
  - **Artifact Saved:** Serialized timestamp-level prediction records ($N = 23,193$ rows) with columns `[timestamp, cohort_type, district, meter_id, actual_demand_kw, predicted_demand_kw, naive_persistence_kw, model_architecture, feature_config, exp_id]` to:
    - `artifacts/metrics/aggregate_test_predictions.parquet`
    - `artifacts/metrics/aggregate_test_predictions.csv`
- **Individual Pooled Panels (Mathura & Bareilly):**
  - **Audit Status:** Documented limitation. 17 of 18 individual checkpoints were trained on Google Colab T4 GPU and currently reside in the Colab session storage. Local disk contains only `ind_mathura_1D_CNN_consumption_only.pt`.
  - **Scientific Integrity Guard:** Per Phase 3.2 guidelines, missing individual predictions were **NOT** fabricated, approximated, or imputed from summary statistics. Instead, the individual panel test protocol is formally documented, the runner script `scripts/generate_individual_predictions.py` is provided for execution upon checkpoint export, and the aggregate DM suite is rigorously analyzed.

---

## 2. Statistical Methodology

### 2.1 Diebold–Mariano Test Formulation
Given observed demand $y_t$ and competing forecasts $\hat{y}_{1, t}$ and $\hat{y}_{2, t}$ at forecast horizon $H = 1$, the loss differential series is defined as:
$$d_t = L(y_t - \hat{y}_{1, t}) - L(y_t - \hat{y}_{2, t}) = L(e_{1, t}) - L(e_{2, t})$$

- **Primary Loss Function:** Squared-Error Loss ($L(e) = e^2$), directly mapping to Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) in physical $\text{kW}^2$.
- **Supplementary Loss Function:** Absolute-Error Loss ($L(e) = |e|$), directly mapping to Mean Absolute Error (MAE) in physical $\text{kW}$.

The null hypothesis of equal predictive accuracy is:
$$H_0: \mathbb{E}[d_t] = 0 \quad \text{vs.} \quad H_1: \mathbb{E}[d_t] \neq 0$$

### 2.2 Heteroskedasticity and Autocorrelation Consistent (HAC) Variance
Because 15-minute load forecast errors exhibit high-frequency temporal autocorrelation and intraday volatility clustering, standard OLS variance estimators are biased. We estimate the asymptotic variance of the sample mean differential $\bar{d} = \frac{1}{T} \sum_{t=1}^T d_t$ using the Newey–West (1987) HAC covariance estimator with Bartlett kernel weights:
$$\hat{V}(\bar{d}) = \frac{1}{T} \left( \hat{\gamma}_0 + 2 \sum_{k=1}^K \left(1 - \frac{k}{K + 1}\right) \hat{\gamma}_k \right)$$
where $\hat{\gamma}_k = \frac{1}{T} \sum_{t=k+1}^T (d_t - \bar{d})(d_{t-k} - \bar{d})$, and truncation lag $K = \lfloor 4 (T/100)^{2/9} \rfloor \approx 8$ lags (2 hours of 15-minute autocorrelation memory).

### 2.3 Finite-Sample Adjustment & Hypothesis Testing
We apply the Harvey, Leybourne, and Newbold (HLN, 1997) finite-sample correction:
$$DM_{\text{HLN}} = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} \times \sqrt{\frac{T + 1 - 2H + H(H-1)/T}{T}}$$
Two-sided $p$-values are computed against Student's $t$-distribution with $T - 1$ degrees of freedom.

### 2.4 Multiple-Comparison Correction (Holm–Bonferroni)
To strictly control the Family-Wise Error Rate (FWER) across the family of 7 pairwise hypotheses per cohort, we apply the Holm–Bonferroni step-down correction:
$$p^{\text{adj}}_{(i)} = \min \left( 1, \max_{j \le i} (m - j + 1) p_{(j)} \right)$$
where $m = 7$ is the number of simultaneous tests per cohort.

---

## 3. Master Pairwise Diebold–Mariano Results

All 14 hypothesis tests across the two aggregate cohorts were conducted using paired chronological observations ($N = 1,320$ for Mathura, $N = 1,257$ for Bareilly; total aggregate test sample count = 2,577).

### Table 3.1: Complete Diebold–Mariano Benchmark Table (Aggregate Demand)

| Cohort | Comparison Hypothesis | Model 1 ($\hat{y}_1$) | Model 2 ($\hat{y}_2$) | Primary DM Stat (SE) | Raw $p$-value | Holm $p_{\text{adj}}$ | FWER Sig ($\alpha=0.05$) | Direction of Superiority | Supp. DM Stat (AE) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Mathura** | **1D CNN vs LSTM** | 1D CNN | LSTM | **+2.5817** | 0.0099 | **0.0099** | **YES** | **LSTM Superior ($p < 0.01$)** | +2.0521 |
| **Mathura** | **LSTM vs CNN–LSTM** | LSTM | CNN–LSTM | **-11.5110** | $< 10^{-15}$ | **$< 10^{-15}$** | **YES** | **LSTM Superior ($p < 10^{-15}$)** | -11.0845 |
| **Mathura** | **1D CNN vs CNN–LSTM** | 1D CNN | CNN–LSTM | **-11.4176** | $< 10^{-15}$ | **$< 10^{-15}$** | **YES** | **1D CNN Superior ($p < 10^{-15}$)** | -10.9632 |
| **Mathura** | **Consumption vs Retro Weather** | LSTM (Cons) | LSTM (Retro) | **-14.2249** | $< 10^{-15}$ | **$< 10^{-15}$** | **YES** | **Consumption Superior ($p < 10^{-15}$)** | -14.6102 |
| **Mathura** | **Consumption vs Operational Weather** | LSTM (Cons) | LSTM (Op) | **-13.1316** | $< 10^{-15}$ | **$< 10^{-15}$** | **YES** | **Consumption Superior ($p < 10^{-15}$)** | -13.4891 |
| **Mathura** | **Retro vs Operational Weather** | LSTM (Retro) | LSTM (Op) | **+10.0710** | $< 10^{-15}$ | **$< 10^{-15}$** | **YES** | **Operational Superior ($p < 10^{-15}$)** | +8.9142 |
| **Mathura** | **Best DL vs Naive Persistence** | LSTM (Cons) | Naive Pers | **+9.9719** | $< 10^{-15}$ | **$< 10^{-15}$** | **YES** | **Persistence Superior ($p < 10^{-15}$)** | +7.8415 |
| **Bareilly** | **1D CNN vs LSTM** | 1D CNN | LSTM | **-5.9129** | $4.33 \times 10^{-9}$ | **$4.33 \times 10^{-9}$** | **YES** | **1D CNN Superior ($p < 10^{-8}$)** | -6.2140 |
| **Bareilly** | **LSTM vs CNN–LSTM** | LSTM | CNN–LSTM | **-7.0331** | $3.31 \times 10^{-12}$ | **$1.32 \times 10^{-11}$** | **YES** | **LSTM Superior ($p < 10^{-10}$)** | -6.8912 |
| **Bareilly** | **1D CNN vs CNN–LSTM** | 1D CNN | CNN–LSTM | **-6.6071** | $5.78 \times 10^{-11}$ | **$1.73 \times 10^{-10}$** | **YES** | **1D CNN Superior ($p < 10^{-9}$)** | -6.4521 |
| **Bareilly** | **Consumption vs Retro Weather** | 1D CNN (Cons) | 1D CNN (Retro) | **-10.4953** | $< 10^{-15}$ | **$< 10^{-15}$** | **YES** | **Consumption Superior ($p < 10^{-15}$)** | -10.8214 |
| **Bareilly** | **Consumption vs Operational Weather** | 1D CNN (Cons) | 1D CNN (Op) | **-11.3210** | $< 10^{-15}$ | **$< 10^{-15}$** | **YES** | **Consumption Superior ($p < 10^{-15}$)** | -11.6450 |
| **Bareilly** | **Retro vs Operational Weather** | 1D CNN (Retro) | 1D CNN (Op) | **-8.8372** | $< 10^{-15}$ | **$< 10^{-15}$** | **YES** | **Retro Superior ($p < 10^{-15}$)** | -8.1209 |
| **Bareilly** | **Best DL vs Naive Persistence** | 1D CNN (Cons) | Naive Pers | **+6.3153** | $3.73 \times 10^{-10}$ | **$7.47 \times 10^{-10}$** | **YES** | **Persistence Superior ($p < 10^{-9}$)** | +5.9421 |

---

## 4. Empirical Interpretation & Core Scientific Findings

### 4.1 Architectural Hierarchy
1. **Mathura Aggregate:** LSTM achieved significantly lower squared error than 1D CNN ($DM = +2.5817, p_{\text{adj}} = 0.0099$). Both standalone architectures vastly outperformed the hybrid CNN–LSTM ($DM = -11.51$ and $-11.42$, $p < 10^{-15}$).
2. **Bareilly Aggregate:** 1D CNN achieved significantly lower squared error than LSTM ($DM = -5.9129, p_{\text{adj}} = 4.33 \times 10^{-9}$). Once again, CNN–LSTM ranked worst ($DM = -7.03$ and $-6.61$, $p < 10^{-9}$).
3. **Hybrid Model Pathology:** In aggregate demand forecasting, the cascaded CNN–LSTM architecture consistently underperformed standalone 1D CNN and LSTM across both districts. Compounding CNN feature pooling with recurrent gates over short 16-step sequences introduced unnecessary model complexity and optimization variance without predictive gain.

### 4.2 Impact of Exogenous Weather Covariates at Feeder Scale
1. **Superiority of Consumption-Only:** In both Mathura and Bareilly aggregate cohorts, **Consumption-Only models demonstrated statistically superior performance over all weather-augmented models ($p < 10^{-15}$)**.
2. **Mechanistic Explanation:** Aggregate feeder load aggregates thousands of diverse domestic appliances across an entire municipal sector, creating a highly autocorrelated, smoothly varying load profile. Adding 9 ERA5-Land reanalysis weather covariates into a 4-hour ($L=16$) sliding window inflated input dimensionality by $10\times$, resulting in empirical overfitting to short-term meteorological fluctuations.
3. **Operational vs. Retrospective Weather:** In Mathura, Operational Weather (with 5-day latency) was statistically superior to Retrospective Weather ($DM = +10.0710, p < 10^{-15}$), whereas in Bareilly, Retrospective was superior ($DM = -8.8372, p < 10^{-15}$). However, both were substantially inferior to simple Consumption-Only.

### 4.3 Deep Learning vs. Naive Persistence (The Resolution Paradox)
1. **At Aggregate Scale:** Naive Persistence ($y_{t+1} = y_t$) was statistically superior to the best deep learning models ($DM = +9.9719$ in Mathura, $+6.3153$ in Bareilly, $p < 10^{-9}$). Due to the continuous inertia of feeder-level load at 15-minute resolution, one-step persistence produces minimal step-change error, making it a very difficult benchmark to beat in pure RMSE on smooth series.
2. **Contrast with Household Scale:** While persistence dominates smooth aggregate feeders at 15-minute horizon, Phase 3.1 demonstrated that at the **individual household meter level**, deep learning decisively beat persistence by **$+20.6\%$ in Mathura** (`ind_mathura_CNN_LSTM_consumption_weather_operational`, $SS_{\text{pers}} = +0.2057$) and **$+14.4\%$ in Bareilly** (`ind_bareilly_LSTM_consumption_weather_operational`, $SS_{\text{pers}} = +0.1440$). The intermittent, stochastic spikes of individual households cannot be tracked by persistence, allowing deep learning to deliver massive empirical gains.

---

## 5. Individual Pooled Panel DM Protocol (Documented Limitation)

### 5.1 Status and Cause
The 18 pooled individual-meter experiments were trained and evaluated on Google Colab T4 GPU. The resulting metrics were captured in `training_results.json` and `master_experiment_comparison.png`. However, the individual prediction arrays ($N = 229,281$ for Mathura, $N = 398,897$ for Bareilly) require either local checkpoint files (`artifacts/checkpoints/ind_*.pt`) or execution on Colab.

### 5.2 Clustered / Panel Dependence Guard
Treating 628,178 pooled household test observations as independent time series samples would produce severe Type I error inflation due to:
- **Cross-sectional spatial correlation** (all households in Mathura experience the same ambient temperature and sunset simultaneously).
- **Within-meter serial autocorrelation** across consecutive 15-minute intervals.

**Methodological Requirement:**
When individual prediction arrays are generated via `scripts/generate_individual_predictions.py`:
- DM tests must either be computed **independently per household meter** (generating a distribution of $M=38$ and $M=46$ DM statistics tested via Wilcoxon signed-rank test), or
- Evaluated via a **cluster-robust block bootstrap** resampling entire household load trajectories to preserve intra-household serial dependence and cross-household spatial dependence.

---

## 6. Generated Artifacts & Deliverables

1. **Prediction Dataset:** `artifacts/metrics/aggregate_test_predictions.parquet` ($N = 23,193$ rows).
2. **Statistical Results Table:** `artifacts/metrics/dm_test_results.csv` (16 test records with test statistics, $p$-values, effect directions, and FWER flags).
3. **Summary JSON:** `artifacts/metrics/dm_test_summary.json`.
4. **Significance Comparison Chart:** `artifacts/metrics/dm_significance_comparison.png` (Horizontal bar plot showing HLN-corrected DM statistics and Holm-adjusted $p$-values against the critical $|DM| = 1.96$ boundary).
5. **Individual Prediction Engine:** `scripts/generate_individual_predictions.py`.

---

## 7. Phase 3.2 Sign-Off & Verification

| Checklist Item | Status | Verification Detail |
| :--- | :---: | :--- |
| **Prediction Artifact Audit** | **PASS** | Mathura ($N=1,320$) & Bareilly ($N=1,257$) continuous intervals verified; RMSEs match logs exactly. |
| **No Model Retraining** | **PASS** | Evaluation performed strictly via pre-trained checkpoints. |
| **Primary Loss (Squared Error)** | **PASS** | Diebold–Mariano computed using physical $\text{kW}^2$. |
| **Supplementary Loss (Absolute Error)**| **PASS** | Diebold–Mariano computed using physical $\text{kW}$. |
| **Newey–West HAC Variance** | **PASS** | Bartlett kernel with $K=8$ lags accounts for 2-hr autocorrelation. |
| **Holm–Bonferroni FWER Control** | **PASS** | Step-down multiplier applied to all tests per cohort. |
| **Scientific Integrity on Pooled Scope** | **PASS** | Documented limitation; no simulated or invented predictions. |

