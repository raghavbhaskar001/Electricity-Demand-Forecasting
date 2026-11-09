# Phase 1.3: Data Integrity and Pre-Processing Verification Test Plan

**Project Title:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Document Revision:** Final Methodology Refinement (Pre-Execution Verification Test Plan)  
**Execution Timestamp:** 2026-11-02T00:05:40+05:30  
**Compliance Notice:** This document defines the pytest test cases, input fixtures, and validation assertions for Phase 1.3. It does not access, read, or execute against files under `data/raw/`.

---

## 1. Overview of Verification Test Suite

The integrity test suite guarantees that all structural assumptions, physical bounds, and aggregation formulas are unit-tested and verified with synthetic micro-fixtures before applying them to the full empirical datasets.

The tests are organized into six distinct functional modules:
1. `test_cadence_and_grid`: Timestamp regularity, monotonicity, and duplicate detection.
2. `test_target_construction`: Exact timestamp-indexed 15-minute aggregation vs. rejection of naive 5-row sums.
3. `test_electrical_physics`: Physical boundary constraints, voltage/current consistency, and power factor checks.
4. `test_missingness_and_gaps`: Outage detection, transition gap boundary isolation, and missingness masks.
5. `test_cohort_aggregation`: Multi-meter synchronous summation, missing-meter scaling, and scientific label enforcement.
6. `test_weather_leakage`: Causal alignment and strict historical masking for weather covariates.

---

## 2. Test Case Specifications

### Module 1: Timestamp Cadence & Temporal Grid (`test_cadence_and_grid.py`)

#### Test Case 1.1: `test_regular_15min_cadence()`
- **Input:** Sample series with timestamps `[2013-08-01 00:15:00, 00:30:00, 00:45:00, 01:00:00]`.
- **Assertion:**
  $$\Delta t = t_k - t_{k-1} = 15\text{ minutes} \quad \forall k$$
- **Pass Criteria:** Zero anomalies, zero negative deltas.

#### Test Case 1.2: `test_duplicate_timestamp_detection()`
- **Input:** Series containing duplicate entry: `[..., 00:30:00, 00:30:00, ...]`.
- **Assertion:** Flag $(m, t)$ collision; ensure deduplication policy retains singular record without corrupting sum.

#### Test Case 1.3: `test_3min_cadence_and_gap_isolation()`
- **Input:** Series with regular 3-min steps followed by a 30-minute blackout: `[..., 03:00, 03:03, 03:06, 03:36, 03:39]`.
- **Assertion:** Identify jump $\Delta t = 30\text{ min} > 3\text{ min}$; verify that the gap interval $(03:06, 03:36]$ is marked as missing and not interpolated.

---

### Module 2: Target Construction & Aggregation (`test_target_construction.py`)

#### Test Case 2.1: `test_strict_5_interval_aggregation_pass()`
- **Input:** Meter fixture with five verified 3-minute sub-intervals in $(00:00, 00:15]$:
  $$\{00:03, 00:06, 00:09, 00:12, 00:15\} \quad \text{with } \text{t\_kWh} = [0.02, 0.02, 0.02, 0.02, 0.02]$$
- **Expected Output:**
  $$\text{Energy}_{15\text{min}} = \sum_{k=1}^5 0.02 = 0.10\text{ kWh}$$
  $$\text{Interval Timestamp} = \text{2019-05-01 00:15:00}$$
  $$\text{Status} = \text{COMPLETE}$$

#### Test Case 2.2: `test_rejection_of_naive_5_row_sum_on_discontinuous_data()`
- **Input:** Five consecutive rows spanning across a 1-hour blackout:
  $$\{10:00, 10:03, 11:00, 11:03, 11:06\}$$
- **Assertion:**
  - Naive 5-row rolling sum would return a false value over 1 hour 6 minutes.
  - **Timestamp-indexed aggregator MUST reject window $(10:00, 10:15]$ as incomplete** ($|\mathcal{O}| = 2 < 5$) and return `NaN` or `INCOMPLETE`.
- **Pass Criteria:** `ValueError` or `MaskedInterval` triggered.

#### Test Case 2.3: `test_iiitd_lnenergy_to_kwh_conversion()`
- **Input:** Sample $\text{lnenergy} = -2.56151775$.
- **Expected Output:**
  $$\text{energy} = \exp(-2.56151775) = 0.0771879\text{ kWh}$$
- **Assertion:** Ensure $\text{energy} > 0$ and $\text{abs}(\text{energy} - 0.077188) < 10^{-6}$.

---

### Module 3: Electrical Physics Validation (`test_electrical_physics.py`)

#### Test Case 3.1: `test_screening_thresholds_vs_deletion()`
- **Input:** Series of phase voltage values: `[242.0, 241.5, 0.0, 295.0, 140.0]`.
- **Assertion:**
  - $V = 0.0\text{ V}$ classified as `IS_OUTAGE = True` (genuine empirical power cut, retained).
  - $V = 295.0\text{ V}$ classified as `IS_OVERVOLTAGE = True` (flagged for screening, **NOT automatically deleted**).
  - $V = 140.0\text{ V}$ classified as `IS_UNDERVOLTAGE_BROWNOUT = True` (flagged, retained).
  - Ensure zero records are silently dropped from the raw table.

#### Test Case 3.2: `test_current_energy_consistency()`
- **Input:** 3-minute interval with $V = 240\text{ V}$, $I = 2.0\text{ A}$, $\Delta t = 0.05\text{ h}$.
- **Expected Apparent Energy:** $S = 240 \times 2.0 \times 0.05 / 1000 = 0.024\text{ kVAh}$.
- **Assertion:** Observed active energy $\text{t\_kWh} \le 0.024\text{ kWh}$ ($\because \text{PF} \le 1.0$). If $\text{t\_kWh} > 0.030\text{ kWh}$, trigger `IS_PHYSICAL_DISCREPANCY = True`.

#### Test Case 3.3: `test_frequency_grid_standards()`
- **Input:** Frequency readings: `[50.02, 49.98, 50.00, 47.2, 53.1]`.
- **Assertion:** Values $47.2\text{ Hz}$ and $53.1\text{ Hz}$ flagged with `IS_FREQUENCY_ANOMALY = True` for diagnostic review.

---

### Module 4: Missingness and Transition Gap Handling (`test_missingness_and_gaps.py`)

#### Test Case 4.1: `test_pctmissing_filtering_iiitd()`
- **Input:** Records with $\text{pctmissing} \in [0, 5, 25, 80, 100]$.
- **Assertion:** Threshold filter $\tau = 10\%$ retains intervals 0 and 5; flags intervals 25, 80, and 100 as degraded reliability.

#### Test Case 4.2: `test_annual_transition_gap_detection_mathura()`
- **Input:** File boundary records between Mathura 2019 tail (`2019-12-31 23:57:00`) and Mathura 2020 head (`2020-01-06 00:00:00`).
- **Assertion:** System detects gap duration $\Delta t = 120.05\text{ hours}$ (5 days, 3 minutes) and treats January 1–5 as missing on master temporal grid.

---

### Module 5: Cohort Aggregation & Label Enforcement (`test_cohort_aggregation.py`)

#### Test Case 5.1: `test_iiitd_ground_truth_and_partial_sum()`
- **Input A (All 19 Apartments Present):**
  - All 19 apartments valid at timestamp $T$.
  - Assertion: Returns valid ground-truth aggregate $Y_{\text{agg}}^{\text{primary}}(T) = \sum_{i=1}^{19} \exp(\text{lnenergy}_{i, T})$.
- **Input B (18 Apartments Present):**
  - Exactly 18 apartments valid at timestamp $T$.
  - Assertion:
    - Primary benchmark returns `NaN` / `INCOMPLETE_COHORT`.
    - Sensitivity pipeline returns observed partial sum labeled `"18-Apartment Observed Partial Sum"`.
    - **Strict Gate:** Fails if the system attempts to impute the 19th apartment's demand and label it measured ground truth.

#### Test Case 5.2: `test_ceew_primary_vs_sensitivity_thresholds()`
- **Input Fixture:** Fixed cohort of 40 smart meters.
- **Scenario A (37 meters active, 92.5%):**
  - Assertion: Passes primary analysis threshold ($\ge 90\%$). Valid primary cohort sum computed.
- **Scenario B (35 meters active, 87.5%):**
  - Assertion: Fails primary threshold ($< 90\%$); passes sensitivity analysis threshold ($\ge 85\%$). Tagged `SENSITIVITY_ONLY`.
- **Scenario C (31 meters active, 77.5%):**
  - Assertion: Fails both primary and sensitivity thresholds ($< 85\%$). Marked `LOW_COVERAGE_EXCLUDED`.

#### Test Case 5.3: `test_scientific_label_enforcement()`
- **Input:** Cohort aggregation pipeline output metadata.
- **Assertion:**
  - Valid strings: `"19-Apartment Housing Cohort Demand"`, `"Mathura Household Sample Cohort Demand"`.
  - Illegal strings that raise `AssertionError`: `"Delhi City Demand"`, `"Mathura Feeder Demand"`, `"Campus Total Demand"`.

---

### Module 6: Weather Causal Alignment & Leakage Prevention (`test_weather_leakage.py`)

#### Test Case 6.1: `test_causal_weather_lagging()`
- **Forecast Origin:** $T = \text{2019-05-01 10:15:00}$.
- **Target to Predict:** Electricity demand at $T + 15\text{min} = \text{10:30:00}$.
- **Weather Availability:** Hourly reanalysis published at `10:00:00` and `11:00:00`.
- **Assertion:**
  - Input tensor at origin $T$ MUST ONLY contain weather timestamped $\le \text{10:00:00}$ (staleness = 15 min).
  - Weather record at `11:00:00` must be strictly inaccessible ($t_{\text{weather}} > T$).
  - Assert `max(input_weather_timestamps) <= T`.

---

## 3. Execution Framework

All tests will be implemented under `tests/test_data_integrity.py` using standard `pytest`:

```powershell
# Command syntax for Phase 1.3 test execution
pytest tests/test_data_integrity.py -v
```

All test fixtures will use programmatic in-memory dummy time-series arrays to guarantee zero coupling or mutation of the raw datasets.
