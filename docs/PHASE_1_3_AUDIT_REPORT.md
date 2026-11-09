# Phase 1.3: Raw Dataset Integrity and Structural Audit Report

**Generated:** 2026-11-02 11:46:02
**Status:** Empirical Validation Completed (Strict Read-Only Mode)
**Compliance:** Non-destructive screening flags, anti-row-shifting target verification, distinct unit-test isolation.

---

## 1. Executive Summary

- **Dataset A (IIIT Delhi Faculty Housing):** Audited `Table_A2_Final.tab` (375,805 records across 19 apartments). Verified exact zero duplicate timestamps across all meters. Logarithmic energy values successfully inverted to physical kWh.
- **Dataset B (CEEW Mathura & Bareilly):** Audited all 6 annual smart meter files across 2019, 2020, and 2021 totaling **21,394,429 telemetry records**.
- **Physical Integrity:** Zero negative energy values ($0.0\%$) and zero unphysical high-load spikes ($>0.5\text{ kWh / 3-min}$) observed across all ~21.4 million records.
- **Operating State Disaggregation:** Explicitly separated genuine electrical outages ($V < 50\text{ V}$, $I = 0\text{ A}$), genuine consumer zero-load intervals ($V \ge 150\text{ V}$, $I = 0\text{ A}$, appliances off), and missing intervals.
- **Target Construction Verification:** Confirmed that raw telemetry must be aggregated via timestamp-indexed 5-interval mapping rather than naive 5-row rolling summation.

---

## 2. Dataset A (IIIT Delhi) Empirical Audit

- **Total 15-Minute Timestamps in Dataset:** 27,161
- **Primary Ground-Truth Timestamps (All 19 Apartments Synchronously Present):** 599 (2.21%)
- **Sensitivity Timestamps (Exactly 18 Apartments Present - Observed Partial Sum):** 2,400 (8.84%)
- **Incomplete Timestamps (<18 Apartments Present):** 24,162 (88.96%)

> [!NOTE]
> Because apartments entered the study at staggered start dates (e.g., Apt 2 in Sept 2013, Apt 14 in Aug 2013), the synchronous 19-apartment primary cohort aggregate is available for a focused subset of 599 intervals. Incomplete intervals must not be imputed and falsely labeled measured ground truth.

### Census Summary Table (First 10 Apartments):

| dataset | entity_id | total_records | start_time | end_time | cadence_15m_pct | duplicate_timestamps | gap_count_gt15m | min_energy_kwh | median_energy_kwh | max_energy_kwh | pctmissing_zero_pct | pctmissing_le10_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IIIT Delhi | Apt_1 | 17912 | 2013-08-01 00:15:00 | 2014-05-12 23:45:00 | 93.91 | 0 | 1091 | 6.2e-05 | 0.0509 | 1.236 | 99.39 | 99.39 |
| IIIT Delhi | Apt_2 | 15614 | 2013-09-01 20:00:00 | 2014-05-12 23:45:00 | 96.0 | 0 | 624 | 1.6e-05 | 0.0645 | 1.5721 | 99.44 | 99.44 |
| IIIT Delhi | Apt_3 | 17038 | 2013-08-01 00:15:00 | 2014-05-12 23:45:00 | 93.64 | 0 | 1084 | 6.3e-05 | 0.0394 | 1.3719 | 98.71 | 98.71 |
| IIIT Delhi | Apt_4 | 20793 | 2013-08-01 00:15:00 | 2014-05-12 23:45:00 | 99.79 | 0 | 43 | 0.000688 | 0.0904 | 1.1796 | 99.4 | 99.4 |
| IIIT Delhi | Apt_5 | 15650 | 2013-08-04 01:15:00 | 2014-05-12 23:45:00 | 98.72 | 0 | 200 | 3.1e-05 | 0.0294 | 2.0708 | 99.44 | 99.44 |
| IIIT Delhi | Apt_6 | 15961 | 2013-08-01 00:15:00 | 2014-05-12 23:45:00 | 95.46 | 0 | 724 | 3.1e-05 | 0.018 | 0.6736 | 99.52 | 99.52 |
| IIIT Delhi | Apt_7 | 24547 | 2013-08-01 00:15:00 | 2014-05-12 23:45:00 | 96.39 | 0 | 886 | 6.3e-05 | 0.0716 | 3.8522 | 99.55 | 99.55 |
| IIIT Delhi | Apt_8 | 23363 | 2013-08-01 00:15:00 | 2014-05-12 23:45:00 | 92.31 | 0 | 1796 | 1.6e-05 | 0.0341 | 1.0519 | 99.53 | 99.53 |
| IIIT Delhi | Apt_9 | 17979 | 2013-08-01 00:15:00 | 2014-05-12 23:45:00 | 75.09 | 0 | 4478 | 8e-06 | 0.0137 | 0.6193 | 99.5 | 99.5 |
| IIIT Delhi | Apt_10 | 24157 | 2013-08-01 00:15:00 | 2014-05-12 23:45:00 | 92.2 | 0 | 1883 | 0.000219 | 0.0323 | 1.1227 | 99.59 | 99.59 |

---

## 3. Dataset B (CEEW Mathura & Bareilly) Empirical Telemetry Audit

### Operating State Disaggregation Definition:
- **Outage (`IS_OUTAGE`):** $V < 50\text{ V}$, $I = 0\text{ A}$, $\text{kWh} = 0.0$ (Grid power failure / blackout).
- **Zero Load (`IS_ZERO_LOAD`):** $V \ge 150\text{ V}$, $I = 0\text{ A}$, $\text{kWh} = 0.0$ (Grid healthy, all household appliances powered off).
- **Brownout / Undervoltage:** $50\text{ V} \le V < 150\text{ V}$.
- **Overvoltage:** $V > 280\text{ V}$.

### Mathura Annual Telemetry Summary:

| district | year | filename | total_rows | distinct_meters | start_time | end_time | outage_records | outage_pct | zero_load_records | zero_load_pct | overvoltage_records | undervoltage_records | high_load_spike_records | neg_energy_records | freq_anomaly_records | v_mean | v_std | i_mean | kwh_mean_3min | freq_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mathura | 2019 | CEEW - Smart meter data Mathura 2019.csv | 3588874 | 38 | 2019-05-01 00:00:00 | 2019-12-31 23:57:00 | 264946 | 7.38 | 127264 | 3.55 | 27425 | 11550 | 0 | 0 | 267646 | 223.4 | 66.3 | 1.94 | 0.0203 | 46.29 |
| mathura | 2020 | CEEW - Smart meter data Mathura 2020.csv | 3759360 | 38 | 2020-01-01 00:00:00 | 2020-12-31 23:57:00 | 417965 | 11.12 | 136632 | 3.63 | 3392 | 3609 | 0 | 0 | 418924 | 215.0 | 78.2 | 1.5 | 0.0154 | 44.44 |
| mathura | 2021 | CEEW - Smart meter data Mathura 2021.tab | 550560 | 35 | 2021-01-01 00:00:00 | 2021-02-20 23:57:00 | 66554 | 12.09 | 12800 | 2.32 | 1649 | 184 | 0 | 0 | 66652 | 223.7 | 83.9 | 0.71 | 0.0069 | 43.96 |

**Annual File Span & Meter-Level Transition Audit (Mathura):**
- **File-Level Span:** Seamless continuity at the dataset boundary across midnight (`2019-12-31 23:57:00` to `2020-01-01 00:00:00`, $\Delta t = 0.05\text{ h} = 3\text{ min}$; and `2020-12-31 23:57:00` to `2021-01-01 00:00:00`, $\Delta t = 0.05\text{ h}$).
- **Meter-Level Disaggregation:**
  - **2019 → 2020 (38 common meters):** Exactly 10 meters exhibit seamless 3-minute continuity across midnight. 28 meters exhibit individual logging gaps $>24\text{ hours}$ (median gap: 168.05 hours / 7 days; max: 960 hours).
  - **2020 → 2021 (35 common meters):** 15 meters exhibit seamless 3-minute continuity. 20 meters exhibit gaps $>24\text{ hours}$ (median gap: 24.05 hours; max: 816 hours).

### Bareilly Annual Telemetry Summary:

| district | year | filename | total_rows | distinct_meters | start_time | end_time | outage_records | outage_pct | zero_load_records | zero_load_pct | overvoltage_records | undervoltage_records | high_load_spike_records | neg_energy_records | freq_anomaly_records | v_mean | v_std | i_mean | kwh_mean_3min | freq_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bareilly | 2019 | CEEW - Smart meter data Bareilly 2019.tab | 2919315 | 46 | 2019-05-09 00:00:00 | 2019-12-31 23:57:00 | 184766 | 6.33 | 206856 | 7.09 | 11936 | 1854 | 0 | 0 | 187931 | 230.4 | 61.8 | 1.39 | 0.0148 | 46.81 |
| bareilly | 2020 | CEEW - Smart meter data Bareilly 2020.csv | 6627360 | 46 | 2020-01-01 00:00:00 | 2020-12-31 23:57:00 | 468013 | 7.06 | 463260 | 6.99 | 18846 | 2361 | 0 | 0 | 470237 | 226.3 | 64.3 | 1.57 | 0.0167 | 46.46 |
| bareilly | 2021 | CEEW - Smart meter data Bareilly 2021.csv | 3948960 | 38 | 2021-01-01 00:00:00 | 2021-10-31 23:57:00 | 343487 | 8.7 | 154765 | 3.92 | 24345 | 4313 | 0 | 0 | 345047 | 219.6 | 70.0 | 1.81 | 0.0194 | 45.64 |

**Annual File Span & Meter-Level Transition Audit (Bareilly):**
- **File-Level Span:** Seamless continuity at the dataset boundary across midnight (`2019-12-31 23:57:00` to `2020-01-01 00:00:00`, $\Delta t = 0.05\text{ h}$; and `2020-12-31 23:57:00` to `2021-01-01 00:00:00`, $\Delta t = 0.05\text{ h}$).
- **Meter-Level Disaggregation:**
  - **2019 → 2020 (46 common meters):** 29 meters (63%) exhibit seamless 3-minute continuity across midnight. 17 meters exhibit gaps $>24\text{ hours}$ (median gap: 0.05 hours; max: 1080 hours).
  - **2020 → 2021 (38 common meters):** 30 meters (79%) exhibit seamless 3-minute continuity. 8 meters exhibit gaps $>24\text{ hours}$ (median gap: 0.05 hours; max: 264 hours).

> [!IMPORTANT]
> **Scientific Methodology Finding:**
> Although files span continuously at the aggregate district level, individual meters start/stop asynchronously across year boundaries. Therefore, training feature matrices must be indexed strictly on a calendar timestamp grid with individual meter masks, rather than chaining raw rows sequentially.

---

## 4. Verification Test Suite vs. Empirical Dataset Audit Distinction

| Evaluation Realm | Target Artifact | Methodology | Result Status |
|---|---|---|---|
| **Unit Test Suite** | `tests/test_data_integrity.py` | 13 synthetic micro-fixtures asserting math and edge-case logic (anti-row-summing, physics bounds, cohort thresholds) | **13 / 13 PASS (0 errors)** |
| **Empirical Dataset Audit** | `docs/audit_results/*.csv` | Read-only scan of 21,770,234 raw records across IIIT Delhi and CEEW | **COMPLETE & VALIDATED** |

---

## 5. Next Phase Readiness: Transition to Weather Acquisition (Phase 1.4)

All raw electricity datasets have passed integrity, physical consistency, and non-destructive screening verification.
The workspace is authorized to proceed to **Phase 1.4: Weather Acquisition & Spatial Alignment** (Open-Meteo ERA5-Land Reanalysis).
