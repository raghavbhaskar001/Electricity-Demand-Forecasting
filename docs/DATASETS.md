# Datasets & Preprocessing Specification

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala

---

## 1. Smart Meter Telemetry Datasets

### 1.1 CEEW Smart Meter Dataset — Mathura & Bareilly (Primary Modelling Cohorts)
- **Source:** Council on Energy, Environment and Water (CEEW), *High frequency smart meter data from two districts in India (Mathura and Bareilly)*, published on Harvard Dataverse, Version 2.0 (`doi:10.7910/DVN/GOCHJH`, CC0 1.0 Universal Public Domain Dedication).
- **Native Resolution:** 3-minute interval energy (`t_kWh`) and diagnostic records across residential consumers in Uttar Pradesh, India (`data/raw/ceew/`).
- **Cohort Filtering (`src/construct_targets.py`):**
  - Each 15-minute window requires all 5 constituent 3-minute sub-intervals (`{T-12, T-9, T-6, T-3, T}`).
  - Only meters achieving sustained temporal completeness across the study period are included in the fixed modelling cohort:
    - **Mathura Cohort:** `38` residential meters (`May 2019 – May 2020`, `35,127` fifteen-minute timestamps in `mathura_features.parquet`, `1,568,487` meter-interval rows in `mathura_individual_meters.parquet`).
    - **Bareilly Cohort:** `46` residential meters (`May 2019 – October 2020`, `49,336` fifteen-minute timestamps in `bareilly_features.parquet`, `2,713,573` meter-interval rows in `bareilly_individual_meters.parquet`).

### 1.2 IIIT Delhi Faculty Housing Dataset (Exploratory Only)
- **Source:** Indraprastha Institute of Information Technology Delhi (IIITD) & UCLA (*Replication Data for: Information Strategies for Energy Conservation: A Field Experiment in India*, Chen, Delmas, Locke, & Singh 2017, *Energy Economics* 68, 215–227), published on Harvard Dataverse, Version 1.0 (`doi:10.7910/DVN/7MEXN4`, CC0 1.0 Universal Public Domain Dedication, datafile `Table_A2_Final.tab`).
- **Status:** While the raw panel spans 19 faculty apartments, enforcing the strict 19/19 synchronous cohort requirement and contiguity filtering in `delhi_features.parquet` leaves only `96` contiguous 16-step sequences. Because a 70/15/15 split with $L = 16$ boundary purging leaves zero valid test sequences, the New Delhi cohort is restricted to exploratory descriptive statistics and excluded from deep learning training.

---

## 2. Meteorological Reanalysis Data (Open-Meteo ERA5-Land)

Hourly meteorological observations were retrieved from the Copernicus Climate Change Service (C3S) **ERA5-Land** reanalysis archive (`0.1° x 0.1°` native grid, `doi:10.24381/cds.e2161bac`, CC BY 4.0) via the Open-Meteo Historical Weather API (`src/acquire_weather.py`) at the approved grid centroids:
- **Mathura:** `27.5220° N, 77.7044° E` (`UTC+05:30 / Asia/Kolkata`, spatial offset `4.46 km`)
- **Bareilly:** `28.3656° N, 79.4779° E` (`UTC+05:30 / Asia/Kolkata`, spatial offset `4.65 km`)
- **New Delhi:** `28.5764° N, 77.2893° E` (`UTC+05:30 / Asia/Kolkata`, spatial offset `3.96 km`)

### Nine Meteorological Variables
1. `temperature_2m` (°C)
2. `relative_humidity_2m` (%)
3. `dew_point_2m` (°C)
4. `surface_pressure` (hPa)
5. `wind_speed_10m` (km/h)
6. `direct_radiation` (W/m²)
7. `diffuse_radiation` (W/m²)
8. `cloud_cover` (%)
9. `precipitation` (mm)

### Temporal Interpolation & Operational Lag Regimes (`scripts/align_weather_and_load.py`)
Hourly ERA5-Land records are resampled to 15-minute intervals using cubic-spline interpolation for smooth thermodynamic state variables (temperature, humidity, dew point, pressure, wind speed) and step/linear interpolation bounded at zero for radiative and precipitation fluxes.

Two distinct meteorological regimes are constructed:
1. **`consumption_weather_retro` (10 features):** Pairs historical load $y_{t-15:t}$ with contemporaneous ERA5-Land weather variables (`retro_*`). This represents an upper-bound hindcast assuming accurate local weather observations at time $t$.
2. **`consumption_weather_operational` (10 features):** Applies a strict **120-hour (5-day) publication lag** (`operational_*`) to reflect the real-world release latency of verified ERA5-Land reanalysis products when live local weather station telemetry or commercial NWP feeds are unavailable.

---

## 3. Reconciled Sequence Census

| Cohort & Scale | Processed Parquet File | Total Rows | Contiguous 16-Step Windows | Train ($70\%$) | Purge Gaps ($2 \times 16$ per series) | Validation ($15\%$) | Test ($15\%$) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Mathura Aggregate** (`38` meters) | `mathura_features.parquet` | 35,127 | **9,005** | 6,303 | 32 | 1,350 | **1,320** |
| **Bareilly Aggregate** (`46` meters) | `bareilly_features.parquet` | 49,336 | **8,585** | 6,009 | 32 | 1,287 | **1,257** |
| **Mathura Individual** (`38` meters) | `mathura_individual_meters.parquet` | 1,568,487 | **1,536,402** | 1,075,464 | 1,216 | 230,441 | **229,281** |
| **Bareilly Individual** (`46` meters) | `bareilly_individual_meters.parquet` | 2,668,842 | **2,668,842** | 1,868,167 | 1,472 | 400,306 | **398,897** |
| **Delhi Aggregate** (Exploratory) | `delhi_features.parquet` | 370 | **96** | — | — | — | **0** |
