# Phase 1.5: Target Construction & Modelling Dataset Preparation Report

**Project Title:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Document Revision:** Final Target Construction, Feature Engineering & EDA Summary  
**Execution Timestamp:** 2026-11-02 11:55 IST  
**Compliance Notice:** All raw datasets under `data/raw/` remain strictly immutable and read-only. No synthetic observations have been generated or imputed.

---

## 1. Executive Summary

Phase 1.5 successfully translates verified raw electricity telemetry and hourly ERA5-Land reanalysis into analysis- and modeling-ready Parquet matrices aligned on a master 15-minute calendar grid.

### Key Milestones Achieved:
1.  **Target Construction Standardized:**
    *   **IIIT Delhi:** Inverted $\text{lnenergy}$ via $\text{kWh} = \exp(\text{lnenergy})$; converted 15-minute energy to average demand via $\text{kW} = 4 \times \text{kWh}$.
    *   **CEEW Mathura & Bareilly:** Aggregated over the exact 5-subinterval set $\mathcal{S}(T) = \{T-12, T-9, T-6, T-3, T\}$ requiring all 5 observations present. Converted to average demand via $\text{kW} = 4 \times \sum_{k=1}^5 \text{t\_kWh}(t_k)$. Naive rolling sums across power cuts were strictly rejected ($0.13\%\text{--}0.31\%$ incomplete windows flagged).
2.  **Operating State Preservation:**
    *   Preserved genuine consumer zero-load intervals ($V \ge 150\text{ V}, I = 0\text{ A}, \text{kWh} = 0 \implies \text{kW} = 0.0$) as valid targets.
    *   Preserved grid outages ($V < 50\text{ V}, I = 0\text{ A}, \text{kWh} = 0$) with dedicated boolean flags.
3.  **Strict Calendar Grid & Sequence Blocking:**
    *   Aligned every meter to continuous 15-minute grids.
    *   Contiguous lookback sequences ($L=16$ steps = 4 hours, horizon $H=1$ step = 15 minutes) were identified strictly within contiguous blocks. **No sliding sequence crosses a missing interval or logging gap.**
4.  **Dual Weather Integration (Retrospective vs. Operational Availability):**
    *   *Retrospective Reanalysis Features:* Causal backward as-of merge ($t_{\text{weather}} \le T$) with staleness tracking ($\le 60\text{ min}$).
    *   *Operational Availability Features:* Enforced documented 5-day publication latency ($t_{\text{weather}} \le T - 120\text{h}$) with observation age tracking ($\approx 120\text{ hours}$).
5.  **Longitudinal Cohort Demarcation:**
    *   IIIT Delhi: 19 apartments.
    *   CEEW Mathura: 38 fixed longitudinal meters.
    *   CEEW Bareilly: 46 fixed longitudinal meters.

---

## 2. Processed Modelling Datasets

All processed matrices are exported to [`data/processed/`](../data/processed/) in Apache Parquet format:

| Dataset / Cohort | File Name | Size (KB) | Row Count (15-min Intervals) | Columns / Features | Contents |
|---|---|---|---|---|---|
| **IIIT Delhi (Individual)** | `delhi_individual_meters.parquet` | 2,256 KB | 27,359 | 20 (`timestamp` + 19 apartments) | 15-minute average power demand (kW) per apartment. |
| **IIIT Delhi (Aggregate)** | `delhi_aggregate.parquet` | 271 KB | 27,359 | 8 columns | Active counts, primary (19/19) & sensitivity (18/19) cohort demand. |
| **IIIT Delhi (Features)** | `delhi_features.parquet` | 567 KB | 27,359 | 38 features | Targets + retrospective & operational weather + cyclical calendar encodings. |
| **Mathura (Individual)** | `mathura_individual_meters.parquet` | 2,393 KB | 63,553 | 39 (`timestamp` + 38 meters) | 15-minute average power demand (kW) per smart meter. |
| **Mathura (Aggregate)** | `mathura_aggregate.parquet` | 597 KB | 63,553 | 8 columns | Fixed cohort coverage ratio, primary ($\ge 90\%$) & sensitivity demand. |
| **Mathura (Features)** | `mathura_features.parquet` | 1,212 KB | 63,553 | 39 features | Targets + retrospective & operational weather + cyclical calendar encodings. |
| **Bareilly (Individual)** | `bareilly_individual_meters.parquet` | 3,792 KB | 87,073 | 47 (`timestamp` + 46 meters) | 15-minute average power demand (kW) per smart meter. |
| **Bareilly (Aggregate)** | `bareilly_aggregate.parquet` | 812 KB | 87,073 | 8 columns | Fixed cohort coverage ratio, primary ($\ge 90\%$) & sensitivity demand. |
| **Bareilly (Features)** | `bareilly_features.parquet` | 1,649 KB | 87,073 | 39 features | Targets + retrospective & operational weather + cyclical calendar encodings. |

---

## 3. Empirical Target Validity & Sequence Availability Census

### Aggregate Cohort Demand Summary:

| Cohort Series | Master Grid Intervals | Valid Intervals | Valid % | Contiguous Sequences ($L=16, H=1$) | Mean Demand (kW) | Std (kW) | Median (kW) | Max Demand (kW) |
|---|---|---|---|---|---|---|---|---|
| **IIIT Delhi Primary (19/19)** | 27,359 | 553 | 2.02% | **96** | 9.97 kW | 4.31 kW | 9.15 kW | 26.87 kW |
| **IIIT Delhi Sensitivity (18/19)**| 27,359 | 2,262 | 8.27% | **229** | 8.57 kW | 4.22 kW | 7.74 kW | 30.48 kW |
| **Mathura Primary ($\ge 90\%$)** | 63,553 | 9,813 | 15.44% | **9,005** | 16.95 kW | 6.54 kW | 16.12 kW | 39.06 kW |
| **Mathura Sensitivity (85–89%)** | 63,553 | 3,721 | 5.85% | **3,082** | 12.40 kW | 5.38 kW | 11.85 kW | 35.35 kW |
| **Bareilly Primary ($\ge 90\%$)** | 87,073 | 9,721 | 11.16% | **8,585** | 11.35 kW | 4.96 kW | 10.74 kW | 37.56 kW |
| **Bareilly Sensitivity (85–89%)**| 87,073 | 10,706 | 12.30% | **9,046** | 13.21 kW | 5.42 kW | 12.48 kW | 36.96 kW |

### Individual Household / Meter Demand Summary:

| Cohort / District | Number of Series | Min Sequences per Meter | Median Sequences per Meter | Max Sequences per Meter | Total Contiguous Sequences ($L=16, H=1$) |
|---|---|---|---|---|---|
| **IIIT Delhi Apartments** | 19 series | 6,345 | 13,321 | 20,632 | **262,680 sequences** |
| **CEEW Mathura Households** | 38 series | 17,714 | 39,198 | 59,485 | **1,536,402 sequences** |
| **CEEW Bareilly Households** | 46 series | 11,394 | 62,182 | 79,256 | **2,668,842 sequences** |
| **Total Individual Meter Sequences** | **103 series** | — | — | — | **4,467,924 sequences** |

> [!IMPORTANT]
> **Modelling Scale Assessment:**
> For individual meter forecasting, the project has over **4.46 million valid, gap-free training sequences**, providing massive empirical depth for deep learning models (1D CNN, LSTM, CNN-LSTM).
> For aggregate demand forecasting, Mathura and Bareilly provide **9,005** and **8,585** contiguous sequences at the strict 90% threshold (and over 12,000 sequences when sensitivity bands are evaluated).

---

## 4. Weather Covariate Feature Engineering

The feature tables combine retrospective reanalysis observations and operational simulations:

```
+---------------------------------------------------------------------------------------------------------------+
| DUAL WEATHER FEATURE MATRIX ARCHITECTURE                                                                      |
+--------------------------+-----------------------+-----------------------------+------------------------------+
| Feature Group            | Latency Rule          | Columns Generated           | Scientific Purpose           |
+--------------------------+-----------------------+-----------------------------+------------------------------+
| Retrospective Reanalysis | Backward As-Of        | retro_temperature_2m,       | Ideal upper-bound benchmark  |
| (Historical Truth)       | (t_weather <= T)      | retro_relative_humidity_2m, | using reanalysis without     |
|                          | Age: 0 to 60 minutes  | retro_dew_point_2m,         | publication latency.         |
|                          |                       | retro_surface_pressure,     |                              |
|                          |                       | retro_wind_speed_10m,       |                              |
|                          |                       | retro_direct_radiation,     |                              |
|                          |                       | retro_diffuse_radiation,    |                              |
|                          |                       | retro_cloud_cover,          |                              |
|                          |                       | retro_precipitation         |                              |
+--------------------------+-----------------------+-----------------------------+------------------------------+
| Operational Availability | 5-Day Lagged As-Of    | operational_temperature_2m, | Realistic operational        |
| (Simulated Deployment)   | (t_weather <= T-120h) | operational_humidity_2m,    | deployment benchmark where   |
|                          | Age: ~120 hours       | ... (all 9 variables)       | ERA5-Land reanalysis is      |
|                          |                       |                             | published with 5-day delay.  |
+--------------------------+-----------------------+-----------------------------+------------------------------+
| Cyclical Calendar        | Deterministic         | hour_sin, hour_cos,         | Continuous cyclical time     |
| Encodings                | (Known at origin T)   | day_sin, day_cos,           | representations for neural   |
|                          |                       | month_sin, month_cos        | network attention/convolutions|
+--------------------------+-----------------------+-----------------------------+------------------------------+
```

---

## 5. Diagnostic EDA Artifacts & Empirical Visualizations

All figures exported to [`artifacts/metrics/eda/`](../artifacts/metrics/eda/):

1.  **Figure 1: Consumption Distributions & Boxplots ([`fig1_consumption_distributions.png`](../artifacts/metrics/eda/fig1_consumption_distributions.png)):**
    *   Displays non-negative, right-skewed load profiles across all cohorts.
    *   Mathura cohort median demand: $16.12\text{ kW}$; Bareilly median demand: $10.74\text{ kW}$; IIIT Delhi median demand: $9.15\text{ kW}$.
2.  **Figure 2: Diurnal Load Profiles ([`fig2_diurnal_profiles.png`](../artifacts/metrics/eda/fig2_diurnal_profiles.png)):**
    *   Demonstrates distinct diurnal curves: morning rise (06:00–09:00), afternoon thermal plateau, and strong evening peak (19:00–22:00 IST).
    *   Weekday vs. weekend shifts clearly visible with delayed morning peaks on weekends.
3.  **Figure 3: Individual Meter Load Diversity ([`fig3_household_load_diversity.png`](../artifacts/metrics/eda/fig3_household_load_diversity.png)):**
    *   Contrasts highly volatile, spiky individual household demand curves against smoothed aggregate cohort demand.
4.  **Figure 4: Weather vs. Demand Relationships ([`fig4_weather_demand_relationships.png`](../artifacts/metrics/eda/fig4_weather_demand_relationships.png)):**
    *   Identifies quadratic temperature-demand relationship (space heating in winter $<15^\circ\text{C}$ and heavy air-conditioning cooling load $>30^\circ\text{C}$).
    *   Highlights positive correlation with direct solar radiation during peak daylight hours.
5.  **Figure 5: Operating State Breakdown ([`fig5_operating_state_breakdown.png`](../artifacts/metrics/eda/fig5_operating_state_breakdown.png)):**
    *   Shows the cohort availability ratio distribution, justifying the $\ge 90\%$ primary and $85\%\text{--}89\%$ sensitivity thresholds.

*Interactive EDA Notebook:* [`notebooks/01_exploratory_data_analysis.ipynb`](../notebooks/01_exploratory_data_analysis.ipynb)

---

## 6. Verification Test Results

Target construction unit tests implemented in [`tests/test_target_construction_phase1_5.py`](../tests/test_target_construction_phase1_5.py):
*   `TestTargetConstructionFormulas` (exp conversion, 4 * kWh conversion): **PASS**
*   `TestTargetWindowIntegrity` (strict 5-subinterval requirement, gap rejection): **PASS**
*   `TestOperatingStatePreservation` (zero-load validation, outage retention): **PASS**
*   `TestCausalAndOperationalWeather` (backward as-of merge, 5-day operational lag): **PASS**
*   `TestContiguousSequenceBlocking` (gap-free sliding sequence generator): **PASS**
*   **Result:** **8 / 8 Tests PASS (100%)**.

---

## 7. Next Phase Readiness: Transition to Model Architecture & Training (Phase 2.0)

With targets, individual matrices, aggregate matrices, weather features, and sequence generators fully validated, the project is ready to transition to **Phase 2.0 (Deep Learning Model Implementation & Benchmark Training)** on Google Colab / Local environment:
*   **Models:** 1D CNN, LSTM, and CNN–LSTM.
*   **Ablation Studies:**
    1. Consumption-only vs. Consumption + Retrospective Weather vs. Consumption + Operational Weather.
    2. Individual Meter vs. Aggregate Cohort Demand.
*   **Metrics:** RMSE, MAE, MAPE, $R^2$, and peak demand error.
