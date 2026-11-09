# Phase 1.3: Raw Dataset Integrity and Structural Audit Specification

**Project Title:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Document Revision:** Final Methodology Refinement (Pre-Execution Specification)  
**Execution Timestamp:** 2026-11-02T00:05:00+05:30  
**Compliance Notice:** This document defines the exact mathematical, statistical, and structural audit protocols for Phase 1.3. It does not access, read, or modify files under `data/raw/`.

---

## 1. Scope & Audit Objectives

Phase 1.3 establishes empirical validation of all acquired raw electricity files before any downstream feature engineering or model training begins. The audit enforces strict data integrity rules:
1. Identify all entity identifiers (meter IDs, apartment IDs) and verify structural continuity.
2. Verify timestamp monotonicity, interval regularity, duplicate records, and omission gaps.
3. Validate electrical measurement semantics, unit scaling, and physical consistency bounds.
4. Establish anomaly-screening thresholds (preserving out-of-range records with diagnostic flags rather than silently deleting them).
5. Distinguish genuine grid outages, missing observations, and sensor anomalies.
6. Formalize timestamp-indexed 15-minute target construction without artificial row-shifting.
7. Formalize distinct, non-overlapping aggregate cohort definitions (strictly separating apartment/household sample sums from utility feeder or municipal grid demand).

---

## 2. Dataset A Audit Protocol (IIIT Delhi Faculty Housing)

**Resource:** `data/raw/iiit_delhi/Table_A2_Final.tab` (ID: 3055617)  
**Documentation:** `data/raw/iiit_delhi/Data Description...pdf` (ID: 3055649)  
**Nature:** Econometric replication panel table (pre-aggregated to 15-minute intervals).

### A. Identifier & Entity Verification
- **Target Identifiers:** `apt` column containing integer apartment numbers $1, 2, \dots, 19$.
- **Audit Tests:**
  - Verify $\text{unique}(\text{apt}) \subseteq \{1, 2, \dots, 19\}$ with exactly 19 distinct series.
  - Check for unrecognized, null, or out-of-range apartment identifiers.
  - Calculate total record count per apartment: $N_i = \sum_{t} \mathbb{I}(\text{apt} = i)$.

### B. Timestamp Regularity & Grid Alignment
- **Expected Resolution:** $\Delta t = 15\text{ minutes}$.
- **Timezone:** Indian Standard Time (IST, UTC+5:30).
- **Interval Bounds:** Timestamps represent interval-closing instant `YYYY-MM-DD hh:mm:ss.000` (`:15`, `:30`, `:45`, `:00`).
- **Audit Tests:**
  - Convert `timestamp` to localized `datetime64[ns, Asia/Kolkata]`.
  - For each apartment $i$, compute successive timestamp deltas: $\delta_{i, t} = t_{k} - t_{k-1}$.
  - Assert that all $\delta_{i, t} = 15\text{ minutes}$. Flag any $\delta_{i, t} \ne 15\text{ min}$ as either a gap ($\delta > 15\text{ min}$) or an anomaly ($\delta < 15\text{ min}$).
  - Detect duplicate timestamps per apartment: count of $(i, t)$ tuples where frequency $> 1$.

### C. Electrical Variable Validation (`lnenergy` $\rightarrow$ `energy`)
- **Metric Definition:** $\text{lnenergy}_{i, t} = \ln(\text{Energy}_{i, t} \text{ [kWh]})$.
- **Audit Tests:**
  - Check range of $\text{lnenergy}$: confirm no infinite values ($-\infty$ occurs if $\text{energy} = 0$, $+\infty$ represents numerical overflow).
  - Reconstruct physical energy:
    $$\text{energy}_{i, t} = \exp(\text{lnenergy}_{i, t})$$
  - Physical boundary checks (Screening Thresholds — Non-destructive):
    - $\text{energy}_{i, t} > 0\text{ kWh}$ (non-zero due to logarithmic formulation).
    - Faculty residential flat nominal peak threshold: $5\text{ kW} \times 0.25\text{ h} = 1.25\text{ kWh / 15-min}$.
    - Anomaly screening threshold: flag $\text{energy}_{i, t} > 3.0\text{ kWh}$ ($> 12\text{ kW}$ sustained load) with `IS_HIGH_LOAD_ANOMALY = True`. Do not delete; preserve for exploratory inspection.

### D. Missingness Semantics (`pctmissing`)
- **Metric Definition:** Percentage of expected 30-second measurement pulses missing in that 15-minute block ($0\% \le \text{pctmissing} \le 100\%$).
- **Audit Tests:**
  - Verify boundary: $\min(\text{pctmissing}) \ge 0$, $\max(\text{pctmissing}) \le 100$.
  - Compute distribution histograms: percentage of intervals with $\text{pctmissing} = 0\%$, $0 < \text{pctmissing} \le 10\%$, and $\text{pctmissing} > 10\%$.
  - Flag intervals where pulse loss degrades measurement reliability.

### E. Cohort Aggregation Specification (IIIT Delhi)
- **Primary Measured Ground-Truth Aggregate:**
  - Requires **all 19 apartments** present and valid at timestamp $t$ ($|\mathcal{A}_t| = 19$, $\text{pctmissing}_{i, t} \le 10\%$):
    $$Y_{\text{agg}}^{\text{IIITD, primary}}(t) = \sum_{i=1}^{19} \exp(\text{lnenergy}_{i, t})$$
- **Sensitivity Analysis Cohort:**
  - Intervals where exactly $18/19$ apartments are valid may be evaluated separately as an **observed partial sum**:
    $$Y_{\text{agg}}^{\text{IIITD, partial}}(t) = \sum_{i \in \mathcal{A}_t, |\mathcal{A}_t|=18} \exp(\text{lnenergy}_{i, t})$$
  - **Strict Ground-Truth Rule:** Never estimate or impute the missing 19th apartment's demand and label it measured ground truth. The primary benchmark is strictly the full-cohort synchronous sum.

---

## 3. Dataset B Audit Protocol (CEEW Mathura & Bareilly Smart Meters)

**Resources:**
- Mathura: `data/raw/ceew/mathura/` (2019, 2020, 2021)
- Bareilly: `data/raw/ceew/bareilly/` (2019, 2020, 2021)
- Documentation: `data/raw/ceew/CEEW -  ReadMe.txt`  
**Nature:** Raw high-frequency (3-minute) smart meter telemetry logs.

### A. Multi-File Ingestion & Schema Conformity
- **Audit Tests:**
  - Verify exact column schemas across all 6 files:
    $$\text{Cols} = \{\text{x\_Timestamp}, \text{t\_kWh}, \text{z\_Avg Voltage (Volt)}, \text{z\_Avg Current (Amp)}, \text{y\_Freq (Hz)}, \text{meter}\}$$
  - Standardize column names for downstream processing:
    - `timestamp`, `energy_kwh`, `voltage_v`, `current_a`, `frequency_hz`, `meter_id`.
  - Maintain distinction between tab-delimited (`.tab`) and comma-delimited (`.csv`) raw files.

### B. Meter Identifier Audit
- **Target Identifiers:**
  - Mathura meters: alphanumeric prefix `MH` (e.g. `MH01` to `MH50`).
  - Bareilly meters: alphanumeric prefix `BR` (e.g. `BR01` to `BR52`).
- **Audit Tests:**
  - Count unique meter IDs per district per year.
  - Build a meter longitudinal presence matrix tracking whether meter $m$ is active across 2019, 2020, and 2021.
  - Identify utility meter replacements, resets, or churn.

### C. 3-Minute Cadence & Timestamp Integrity
- **Expected Resolution:** Nominal $\Delta t = 3\text{ minutes}$.
- **Timezone:** Indian Standard Time (IST, UTC+5:30).
- **Audit Tests:**
  - For each individual meter $m$, sort chronologically and compute:
    $$\delta_{m, k} = t_{m, k} - t_{m, k-1}$$
  - Identify exact breakdown of deltas:
    - Normal polling: $\delta = 3\text{ min}$.
    - Communication drops / blackouts: $\delta \in \{6\text{ min}, 9\text{ min}, \dots, > 60\text{ min}\}$.
    - Duplicate timestamps: $\delta = 0\text{ min}$.
  - Duplicate resolution policy: If identical $(m, t)$ records exist with identical telemetry, retain 1; if telemetry differs, log collision and compute mean with `IS_DEDUPLICATED = True`.

### D. Annual File Transition Audits
- **Empirical File Transitions to Audit:**
  - **Mathura 2019 $\rightarrow$ 2020:** Document the known gap between `2019-12-31 23:57:00` and `2020-01-06 00:00:00` (5-day gap).
  - **Mathura 2020 $\rightarrow$ 2021:** Document year-end boundary and verify continuity into February 2021.
  - **Bareilly 2019 $\rightarrow$ 2020:** Verify continuity across midnight `2019-12-31 22:06` to `2020-01-01 00:00`.
  - **Bareilly 2020 $\rightarrow$ 2021:** Document the 1-day transition gap (`2020-12-31 22:03` to `2021-01-02 00:00`).
  - Document the exact closing dates in 2021 for both districts.

### E. Electrical Variable Validation & Anomaly Screening Rules
- **Non-Destructive Principle:** Voltage, current, and frequency screening ranges are **anomaly-screening thresholds, not automatic deletion criteria**. Out-of-range records are preserved in the raw tables and explicitly flagged.
- **Categorization of Anomalies vs. Outages:**
  1. **Genuine Grid Outage / Power Cut:**
     - Condition: $V = 0.0\text{ V}$ or $V < 50\text{ V}$, $I = 0.0\text{ A}$, $\text{t\_kWh} = 0.000\text{ kWh}$.
     - Flag: `IS_OUTAGE = True`. (Represents valid empirical power cut).
  2. **Missing Observation / Telemetry Loss:**
     - Condition: Expected 3-minute timestamp is completely absent from the file.
     - Flag: `IS_MISSING_TIMESTAMP = True`.
  3. **Sensor & Grid Stress Anomalies:**
     - Active Energy: $\text{t\_kWh} < 0.000\text{ kWh}$ (flag `IS_NEGATIVE_ENERGY`), or $\text{t\_kWh} > 0.50\text{ kWh / 3-min}$ ($>10\text{ kW}$ load; flag `IS_HIGH_LOAD_SPIKE`).
     - Voltage Bounds: $V \in [150\text{ V}, 280\text{ V}]$ is acceptable operating range. $V > 280\text{ V}$ flagged `IS_OVERVOLTAGE`; $50\text{ V} \le V < 150\text{ V}$ flagged `IS_UNDERVOLTAGE_BROWNOUT`.
     - Current Bounds: $I < 0.0\text{ A}$ flagged `IS_NEGATIVE_CURRENT`; $I > 60.0\text{ A}$ flagged `IS_OVERCURRENT`.
     - Frequency Bounds: IEGC normal band is $49.90\text{ Hz} \le f \le 50.05\text{ Hz}$. Severe grid trip limits are $48.5\text{ Hz} \le f \le 51.5\text{ Hz}$. Values outside $[48.5, 51.5]\text{ Hz}$ flagged `IS_FREQUENCY_ANOMALY`.

---

## 4. Section C: 15-Minute Target Construction Specification

### The "No Naive Summation" Rule
Under no circumstances shall five arbitrary consecutive CSV rows be summed. If rows are missing due to power cuts, a naive row rolling sum would aggregate across temporal gaps, creating invalid targets.

### Exact Mathematical Specification:
1. **Regular Temporal Grid Definition:**
   For a given district or meter, generate a master reference temporal grid $\mathcal{T}_{15}$ spanning from start timestamp $T_{\text{start}}$ to end timestamp $T_{\text{end}}$ at strict 15-minute intervals:
   $$\mathcal{T}_{15} = \{T_0, T_0 + 15\text{min}, T_0 + 30\text{min}, \dots\}$$
2. **Sub-Interval Membership:**
   Each 15-minute target labeled with interval-closing timestamp $T \in \mathcal{T}_{15}$ corresponds to the exact half-open time interval:
   $$\mathcal{W}(T) = (T - 15\text{min}, T]$$
   Containing the theoretical set of five 3-minute sub-intervals:
   $$\mathcal{S}(T) = \{T - 12\text{min}, T - 9\text{min}, T - 6\text{min}, T - 3\text{min}, T\}$$
3. **Completeness & Aggregation Criteria:**
   Let $\mathcal{O}_m(T) \subseteq \mathcal{S}(T)$ be the set of timestamps observed for meter $m$.
   - **Case 1 (Strict Complete Window: $|\mathcal{O}_m(T)| = 5$):**
     $$\text{Energy}_{15\text{min}}(m, T) = \sum_{k=1}^{5} \text{t\_kWh}_m(t_k)$$
     $$\text{Voltage}_{15\text{min}}(m, T) = \frac{1}{5} \sum_{k=1}^{5} V_m(t_k), \quad \text{Current}_{15\text{min}}(m, T) = \frac{1}{5} \sum_{k=1}^{5} I_m(t_k)$$
     $$\text{Frequency}_{15\text{min}}(m, T) = \frac{1}{5} \sum_{k=1}^{5} f_m(t_k)$$
   - **Case 2 (Incomplete Window: $|\mathcal{O}_m(T)| < 5$):**
     - Target is flagged as `incomplete`.
     - *Default Policy:* Strict rejection; window $T$ for meter $m$ is marked as `NaN` in training tensors to eliminate synthetic data creation.

---

## 5. Section D: Aggregate Cohort Methodology

```
+-------------------------------------------------------------------------------------------------------------------------------+
| SCIENTIFIC COHORT AGGREGATION PROTOCOLS                                                                                      |
+--------------------------+------------------------------+---------------------------+-----------------------------------------+
| Dimension                | IIIT Delhi Cohort            | CEEW Mathura Cohort       | CEEW Bareilly Cohort                    |
+--------------------------+------------------------------+---------------------------+-----------------------------------------+
| Fixed Cohort Definition  | Fixed 19 apartments          | Filtered stable meters    | Filtered stable meters                  |
|                          | ($\mathcal{C}_A = \{1..19\}$)| ($\mathcal{C}_M^* \subset \{\text{MHxx}\}$) | ($\mathcal{C}_B^* \subset \{\text{BRxx}\}$)  |
| Primary Analysis         | All 19 apartments valid      | At least 90% of cohort    | At least 90% of cohort                  |
| Threshold                | ($19/19$ complete coverage)  | meters reporting in window| meters reporting in window               |
| Sensitivity Analysis     | Exactly 18 apartments valid  | 85% to <90% of cohort     | 85% to <90% of cohort                   |
| Threshold                | (Reported as partial sum)    | meters reporting in window| meters reporting in window               |
| Missing-Meter Ground     | STRICTLY PROHIBITED:         | Reported as normalized    | Reported as normalized                  |
| Truth Estimation         | Never estimate 19th apt as GT| sample partial sum        | sample partial sum                      |
| Mandatory Scientific     | "19-Apartment Housing Cohort | "Mathura Household Sample | "Bareilly Household Sample              |
| Label                    | Demand"                      | Cohort Demand"            | Cohort Demand"                          |
| Prohibited Labels        | "Campus Load", "Delhi Load"  | "City Load", "Feeder Load"| "City Load", "Feeder Load"              |
+--------------------------+------------------------------+---------------------------+-----------------------------------------+
```

### Detailed Mathematical Formulations:

1. **IIIT Delhi Apartment Cohort Aggregate:**
   - *Primary Measured Ground Truth ($|\mathcal{A}_t| = 19$):*
     $$Y_{\text{agg}}^{\text{IIITD, primary}}(t) = \sum_{i=1}^{19} \exp(\text{lnenergy}_{i, t})$$
   - *Sensitivity Observed Partial Sum ($|\mathcal{A}_t| = 18$):*
     $$Y_{\text{agg}}^{\text{IIITD, sensitivity}}(t) = \sum_{i \in \mathcal{A}_t} \exp(\text{lnenergy}_{i, t})$$
2. **CEEW Household Sample Cohort Aggregate:**
   - *Primary Analysis ($\ge 90\%$ cohort active):*
     $$Y_{\text{agg}}^{\text{CEEW, primary}}(T) = \sum_{m \in \mathcal{C}_{\text{active}}(T)} \left( \sum_{k=1}^{5} \text{t\_kWh}_m(t_k) \right) \quad \text{where } \frac{|\mathcal{C}_{\text{active}}(T)|}{|\mathcal{C}^*|} \ge 0.90$$
   - *Sensitivity Analysis ($85\% \le \text{coverage} < 90\%$):*
     Evaluated separately with explicit partial-cohort notation.

---

## 6. Section E: Quality Assurance Test Suite Specifications

1. **Audit Metric Summary Tables:**
   - Table 1: Entity census (ID, total rows, start date, end date, missing row count, completeness %).
   - Table 2: Physical value distribution (mean, std, min, 1%, 25%, 50%, 75%, 99%, max for energy, V, I, f).
   - Table 3: Anomaly frequency table (counts of `IS_OUTAGE`, `IS_OVERVOLTAGE`, `IS_UNDERVOLTAGE`, `IS_HIGH_LOAD_SPIKE`).
   - Table 4: Cadence and transition gaps (exact timestamp pairs bounding missing intervals $> 15\text{ min}$).
2. **Diagnostic Visualizations (to be saved in `artifacts/metrics/`):**
   - *Figure 1: Missingness & Outage Heatmap.* Meters on Y-axis, time blocks on X-axis, showing packet drops vs. zero-voltage power cuts.
   - *Figure 2: Timestamp Cadence Histogram.* Distribution of $\Delta t$ showing clean spike at 3 minutes and blackout tails.
   - *Figure 3: Aggregate Cohort vs Individual Meter Profiles.* 15-minute demand curve comparing volatile individual households against smoothed cohort sums.
3. **Automated Verification Test Suite:**
   - Implemented in `tests/test_data_integrity.py` asserting non-negativity, grid regularity, cohort bounds, and strict timestamp-indexed aggregation logic.
