# Phase 1.4: Weather Temporal Coverage and Variable Summary Report

**Generated:** 2026-11-02 11:43:09
**Cadence:** Exactly 1-hour intervals (monotonic, 0 duplicates)
**Timezone:** Indian Standard Time (IST, UTC+5:30)
**Operational Availability Notice:** ERA5-Land is a historical reanalysis dataset with ~5-day publication latency. In practical forecasting deployment, inputs at forecast origin T are subject to this latency constraint.

---

## 1. Temporal Coverage Summary

| Location | Cohort | Requested Range | Actual Returned Range | Total Hours | Missing Records | Cadence Regularity |
|---|---|---|---|---|---|---|
| **Delhi** | IIIT Delhi | 2013-08-01 to 2014-05-31 | `2013-08-01 00:00:00` to `2014-05-31 23:00:00` | **7,296** | 0 (100% complete) | Exactly 60.0 min (0 jumps) |
| **Mathura** | CEEW Mathura | 2019-05-01 to 2021-02-28 | `2019-05-01 00:00:00` to `2021-02-28 23:00:00` | **16,080** | 0 (100% complete) | Exactly 60.0 min (0 jumps) |
| **Bareilly** | CEEW Bareilly | 2019-05-01 to 2021-10-31 | `2019-05-01 00:00:00` to `2021-10-31 23:00:00` | **21,960** | 0 (100% complete) | Exactly 60.0 min (0 jumps) |

---

## 2. Meteorological Variables and Physical Ranges

### Delhi (IIIT Delhi)

| Variable Name | Open-Meteo Parameter | Unit | Min | Mean | Max | Std | Missing Count |
|---|---|---|---|---|---|---|---|
| Temperature 2M | `temperature_2m` | °C | 3.3 | 22.28 | 42.3 | 7.55 | 0 |
| Relative Humidity 2M | `relative_humidity_2m` | % | 9.0 | 65.75 | 100.0 | 22.07 | 0 |
| Dew Point 2M | `dew_point_2m` | °C | 1.0 | 14.37 | 27.0 | 6.24 | 0 |
| Surface Pressure | `surface_pressure` | hPa | 973.5 | 987.43 | 1002.0 | 5.6 | 0 |
| Wind Speed 10M | `wind_speed_10m` | m/s | 0.0 | 2.33 | 7.21 | 1.03 | 0 |
| Direct Radiation | `direct_radiation` | W/m² | 0.0 | 150.29 | 860.0 | 226.23 | 0 |
| Diffuse Radiation | `diffuse_radiation` | W/m² | 0.0 | 57.56 | 406.0 | 74.24 | 0 |
| Cloud Cover | `cloud_cover` | % | 0.0 | 31.03 | 100.0 | 37.46 | 0 |
| Precipitation | `precipitation` | mm | 0.0 | 0.06 | 5.6 | 0.33 | 0 |

### Mathura (CEEW Mathura)

| Variable Name | Open-Meteo Parameter | Unit | Min | Mean | Max | Std | Missing Count |
|---|---|---|---|---|---|---|---|
| Temperature 2M | `temperature_2m` | °C | 1.6 | 24.73 | 45.1 | 8.24 | 0 |
| Relative Humidity 2M | `relative_humidity_2m` | % | 8.0 | 64.87 | 100.0 | 22.17 | 0 |
| Dew Point 2M | `dew_point_2m` | °C | -2.5 | 16.4 | 28.2 | 6.86 | 0 |
| Surface Pressure | `surface_pressure` | hPa | 970.9 | 988.32 | 1004.0 | 7.06 | 0 |
| Wind Speed 10M | `wind_speed_10m` | m/s | 0.0 | 2.35 | 9.64 | 1.15 | 0 |
| Direct Radiation | `direct_radiation` | W/m² | 0.0 | 121.2 | 730.0 | 181.12 | 0 |
| Diffuse Radiation | `diffuse_radiation` | W/m² | 0.0 | 76.75 | 415.0 | 96.03 | 0 |
| Cloud Cover | `cloud_cover` | % | 0.0 | 34.34 | 100.0 | 38.0 | 0 |
| Precipitation | `precipitation` | mm | 0.0 | 0.07 | 26.8 | 0.48 | 0 |

### Bareilly (CEEW Bareilly)

| Variable Name | Open-Meteo Parameter | Unit | Min | Mean | Max | Std | Missing Count |
|---|---|---|---|---|---|---|---|
| Temperature 2M | `temperature_2m` | °C | 2.3 | 24.75 | 42.4 | 7.32 | 0 |
| Relative Humidity 2M | `relative_humidity_2m` | % | 7.0 | 70.02 | 100.0 | 21.59 | 0 |
| Dew Point 2M | `dew_point_2m` | °C | -7.1 | 17.84 | 29.2 | 6.88 | 0 |
| Surface Pressure | `surface_pressure` | hPa | 971.2 | 987.56 | 1003.8 | 6.45 | 0 |
| Wind Speed 10M | `wind_speed_10m` | m/s | 0.0 | 2.55 | 9.93 | 1.34 | 0 |
| Direct Radiation | `direct_radiation` | W/m² | 0.0 | 130.39 | 776.0 | 196.59 | 0 |
| Diffuse Radiation | `diffuse_radiation` | W/m² | 0.0 | 74.14 | 421.0 | 91.64 | 0 |
| Cloud Cover | `cloud_cover` | % | 0.0 | 36.48 | 100.0 | 39.59 | 0 |
| Precipitation | `precipitation` | mm | 0.0 | 0.12 | 35.8 | 0.66 | 0 |
