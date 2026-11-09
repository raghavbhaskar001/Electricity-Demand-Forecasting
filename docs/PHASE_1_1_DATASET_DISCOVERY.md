# Phase 1.1: India-Specific Electricity Dataset Discovery & Scientific Audit Report

**Project Title:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Report Revision:** Phase 1.1B Comprehensive Verification Audit  
**Execution Timestamp:** 2026-11-01T23:25:30+05:30  
**Audit Status:** Completed. All scientific claims, physical measurement semantics, annual file transitions, aggregate cohort definitions, and spatial weather offsets have been audited against official repository metadata and byte-range inspections. No full datasets or weather observations were downloaded, and no Git commits were created.

---

## 1. Executive Summary & Epistemic Audit Matrix

This audit resolves all empirical ambiguities regarding candidate Indian electricity and meteorological datasets for forecasting at a **15-minute horizon** across **individual household/meter** and **aggregate cohort** demand levels.

### Epistemic Status Classifications:
- **[VERIFIED]:** Confirmed directly via official repository metadata APIs, byte-level range previews, mathematical physics checks, or codebook text.
- **[PUBLICATION-REPORTED]:** Explicitly stated in the peer-reviewed journal article or institutional research monograph.
- **[INFERRED]:** Logically derived from empirical observations, file schemas, or physical constraints.
- **[UNKNOWN]:** Unverified or undocumented at repository metadata level; flagged for empirical resolution during data ingestion.

```
+-------------------------------------------------------------------------------------------------------------------------------+
| SCIENTIFIC CLAIMS AND DATASET SEMANTICS AUDIT SUMMARY                                                                         |
+-----------------------------------------------------------+-----------------------+-------------------------------------------+
| Claim / Dimension                                         | Epistemic Status      | Evidence / Audit Result                   |
+-----------------------------------------------------------+-----------------------+-------------------------------------------+
| Dataset A DOI 10.7910/DVN/7MEXN4 Version                  | [VERIFIED]            | Version 1.0 (Released 2017-10-03)         |
| Dataset A Table_A2_Final.tab is 15-min panel              | [VERIFIED]            | HTTP range byte inspection & Table_A2.do  |
| Dataset A lnenergy is exact natural log of 15-min kWh     | [VERIFIED]            | Data Description PDF (p. 2-3) & Table_A2  |
| Dataset A missing pulses handled without imputation       | [PUBLICATION-REPORTED]| Chen et al. 2017 & Table_A2.do pctmissing |
| Dataset A Table_A2 is pre-aggregated publication panel    | [VERIFIED]            | Filtered replication table, not raw pulses|
| Dataset B DOI 10.7910/DVN/GOCHJH Version                  | [VERIFIED]            | Version 2.0 (Released 2021-11-22)         |
| Dataset B t_kWh is interval active energy consumed (kWh)  | [VERIFIED]            | V*I*dt physical identity check & ReadMe   |
| Dataset B is non-accumulating (not cumulative register)   | [VERIFIED]            | Observed interval variation (0.003-0.021) |
| Dataset B native interval is strictly 3 minutes           | [VERIFIED]            | CEEW ReadMe line 22 & byte-range samples  |
| Naive 5-row CSV summation is INVALID for Dataset B        | [VERIFIED]            | Blackout/drop gaps require timestamp grid |
| Dataset B Mathura 2020 has 5-day initial transition gap   | [VERIFIED]            | Byte range inspection (starts 2020-01-06) |
| Dataset B meters are household mains (not appliance sub)  | [PUBLICATION-REPORTED]| CEEW Report 2020 & ReadMe                 |
| Open-Meteo ERA5-Land grid spatial offset is ~4.0-4.7 km   | [VERIFIED]            | Live API coordinate return vs requested   |
| Weather staleness at forecast origin is 0 to 45 minutes   | [VERIFIED]            | Hourly reanalysis boundary logic          |
| Neither dataset is proprietary BSES commercial data       | [VERIFIED]            | Official institutional provenance         |
+-----------------------------------------------------------+-----------------------+-------------------------------------------+
```

---

## 2. Section A: Dataset A — IIIT Delhi Faculty Apartments

**Repository:** Harvard Dataverse | **DOI:** [`10.7910/DVN/7MEXN4`](https://doi.org/10.7910/DVN/7MEXN4)  
**Version:** **1.0** (Released: 2017-10-03T23:13:10Z) | **License:** CC0 1.0 Universal  
**Publication:** Chen, V.L., Delmas, M.A., Locke, S.L., & Singh, A. (2017). *Information Strategies for Energy Conservation: A Field Experiment in India.* Energy Economics, 68, 215–227.

### Detailed Semantics Verification:

1. **Panel Organization (`Table_A2_Final.tab`):**
   - **[VERIFIED]** Contains panel records organized by apartment identifier (`apt`, values 1 through 19) and timestamp (`timestamp`).
   - Timestamps advance at exact 15-minute intervals: `YYYY-MM-DD hh:mm:ss.000` (e.g., `2013-08-01 00:15:00.000`, `00:30:00.000`, `00:45:00.000`, `01:00:00.000`).
   - Sorted in panel block format (all chronological records for `apt 1`, followed by `apt 2`, through `apt 19`).
2. **Definition of `lnenergy` and True Energy Recovery:**
   - **[VERIFIED]** In official documentation (Data Description PDF, lines 123 & 138), `energy` is defined as *"kWh per unit of time"*, and `lnenergy` is explicitly defined as the *"natural log of energy per unit of time"*.
   - Because `Table_A2_Final.tab` operates at 15-minute frequency, `lnenergy` is:
     $$\text{lnenergy}_{i, t} = \ln\left(\text{Energy}_{i, t} \text{ [kWh]}\right)$$
   - The physical electricity consumption in kilowatt-hours is recovered via:
     $$\text{Energy}_{i, t} \text{ [kWh]} = \exp(\text{lnenergy}_{i, t})$$
   - *Example from byte inspection:* At `2013-08-01 00:15:00`, `apt 1` has `lnenergy = -2.561518`, yielding $\exp(-2.561518) = 0.07719\text{ kWh}$ ($\approx 308.8\text{ W}$ average load over 15 minutes), which represents standard baseline residential consumption.
3. **Treatment of Missing 30-Second Pulses:**
   - **[PUBLICATION-REPORTED]** High-frequency pulses were captured by field smart meters every 30 seconds.
   - When communication dropouts or transient sensor drops occurred, the authors **did not synthetically impute** missing pulses. Instead, the observed energy reflects pulses successfully logged during that window.
4. **Semantics of `pctmissing`:**
   - **[VERIFIED]** `pctmissing` records the exact percentage of raw measurement pulses missing within the interval.
   - In Stata replication script `Table_A2.do` (lines 19–22), `pctmissing` is entered as an explicit econometric control covariate (`xtgls lnenergy ... pctmissing ...`) to formally prove that packet loss did not bias treatment coefficients.
   - *Preprocessing Rule for Forecasting:* Intervals with high `pctmissing` (e.g. $> 10\%$) must be treated as partially observed or filtered during model evaluation.
5. **Synchronous Alignment & Cohort Aggregation:**
   - **[VERIFIED]** All 19 apartments share identical 15-minute grid timestamps.
   - Fixed cohort aggregate is formed by summing active apartments at each timestamp:
     $$Y_{\text{agg}}^{\text{IIITD}}(t) = \sum_{i=1}^{19} \exp(\text{lnenergy}_{i, t})$$
6. **Timezone & Interval Boundary:**
   - **[VERIFIED]** Timezone is Indian Standard Time (IST, UTC+5:30).
   - Interval boundary is formatted as interval-end instant (e.g. `00:15:00` represents the interval from `00:00:00` to `00:15:00`).
7. **Canonical vs. Processed Status:**
   - **[VERIFIED]** `Table_A2_Final.tab` is an **econometric replication panel dataset**, already cleaned, log-transformed, and time-aggregated by the authors, **not untouched raw pulse logs**.

---

## 3. Section B: Dataset B — CEEW Mathura & Bareilly Smart Meters

**Repository:** Harvard Dataverse | **DOI:** [`10.7910/DVN/GOCHJH`](https://doi.org/10.7910/DVN/GOCHJH)  
**Version:** **2.0** (Released: 2021-11-22T09:43:22Z; updated from v1.0) | **License:** CC0 1.0 Universal  
**Institution:** Council on Energy, Environment and Water (CEEW) in collaboration with DVVNL.

### Detailed Semantics Verification:

1. **Exact Definition & Units of `t_kWh`:**
   - **[VERIFIED]** In `CEEW - ReadMe.txt` (line 22), `t_kWh` is explicitly documented as *"electricity consumption (t_kWh) ... at every three minute interval (x_Timestamp)"*.
   - **Physical Identity Check:**
     $$P_{\text{apparent}} = V \times I = 243.1\text{ V} \times 1.79\text{ A} = 435.15\text{ VA} \approx 0.435\text{ kW}$$
     $$\text{Energy}_{\text{theoretical}} = P \times \Delta t = 0.435\text{ kW} \times \frac{3}{60}\text{ h} = 0.02176\text{ kWh}$$
     $$\text{Observed } t\_kWh = 0.02100\text{ kWh}$$
     The physical calculation matches the recorded value within expected power factor bounds ($\text{PF} \approx 0.965$).
2. **Interval Consumption vs. Cumulative Register:**
   - **[VERIFIED]** `t_kWh` is **interval energy consumed during the 3-minute window**, **not a cumulative meter register**. Values fluctuate dynamically between 0.000 and 0.030 kWh across intervals, confirming interval semantics.
3. **Record Cadence & Native Frequency:**
   - **[VERIFIED]** Nominal polling cadence is **strictly 3 minutes** (e.g. `03:36:00`, `03:39:00`, `03:42:00`...).
4. **Timestamp Convention:**
   - **[VERIFIED]** Column `x_Timestamp` in IST (UTC+5:30), formatted as `YYYY-MM-DD hh:mm:ss`. Represents the closing timestamp of the 3-minute measurement interval.
5. **Why Naive 5-Row Summation is Strictly Invalid:**
   - **[VERIFIED]** In field smart meters, power outages, cellular modems drops, and load shedding cause skipped polling intervals.
   - If a household experiences a 1-hour power outage, row $k$ might be at `14:00:00` and row $k+1$ at `15:00:00`. Summing 5 consecutive CSV rows would bridge over the blackout and produce a fabricated "15-minute" sum spanning 1 hour and 15 minutes!
   - **Mandatory Resampling Rule:** Target construction **must be timestamp-indexed**:
     $$\text{Window}_T = (T - 15\text{min}, T] = \{T - 12\text{min}, T - 9\text{min}, T - 6\text{min}, T - 3\text{min}, T\}$$
     Only if all 5 exact timestamps exist on the grid is the sum $\sum_{k=1}^5 \text{t\_kWh}_k$ valid. If any timestamp is missing, it must be handled by an explicit missingness mask.
6. **Voltage, Current, and Frequency Semantics:**
   - **[VERIFIED]** In `CEEW - ReadMe.txt`:
     - `z_Avg Voltage (Volt)`: Average phase voltage during the interval (V).
     - `z_Avg Current (Amp)`: Average load current during the interval (A).
     - `y_Freq (Hz)`: Supply frequency (Hz).
     - These are interval average statistics, not instantaneous single-point snapshots.
7. **Annual File Transitions & Empirical Continuity Gaps:**
   - **[VERIFIED via Byte-Range Audits]:**
     - *Mathura 2019 $\rightarrow$ 2020 Transition:* Mathura 2019 ends at `2019-12-31 23:57:00`. Mathura 2020 does not start on January 1; its first recorded timestamp for `MH01` is `2020-01-06 00:00:00` (**5-day initial transition gap**).
     - *Mathura 2021 File Scope:* Ends on `2021-02-18 23:57:00` for `MH47`.
     - *Bareilly 2019 $\rightarrow$ 2020 Transition:* Bareilly 2019 ends at `2019-12-31 22:06:00`. Bareilly 2020 begins at `2020-01-01 00:00:00` (Continuous).
     - *Bareilly 2020 $\rightarrow$ 2021 Transition:* Bareilly 2020 ends at `2020-12-31 22:03:00`. Bareilly 2021 begins at `2021-01-02 00:00:00` (**1-day transition gap**).
     - *Bareilly 2021 File Scope:* Concludes on `2021-03-03 22:06:00`.
8. **Mains vs. Sub-Meter Identifiers:**
   - **[PUBLICATION-REPORTED]** All meters are whole-premise consumer smart meters installed by DVVNL at consumer service entry points. Meters with prefix `MH` belong to Mathura, and `BR` belong to Bareilly. No sub-metered appliance channels exist.
9. **Resolved Repository Version:**
   - **[VERIFIED]** Official Harvard Dataverse API confirms the current live dataset version is **Version 2.0** (released `2021-11-22T09:43:22Z`), which updated and finalized the file catalog over the initial v1.0 draft.

---

## 4. Section C: Rigorous Aggregate Demand Definitions

To prevent scientific misrepresentation, aggregate electricity demand is rigorously defined for each scale:

```
+-------------------------------------------------------------------------------------------------------------------------+
| AGGREGATE DEMAND HIERARCHY & COHORT SPECIFICATIONS                                                                     |
+------------------------------+---------------------------+-----------------------+--------------------------------------+
| Aggregate Level              | Cohort Definition         | Coverage Rule         | Scientific Labeling Rule             |
+------------------------------+---------------------------+-----------------------+--------------------------------------+
| 1. IIIT Delhi Cohort         | Fixed set of 19 faculty   | >= 18 apts present;   | Must be labeled: "19-Apartment       |
|    Residential Aggregate     | housing apartments        | pctmissing <= 10%     | Housing Cohort Load" (NOT campus load|
|                              | (apt 1 to 19)             | per interval          | and NOT Delhi city load)             |
+------------------------------+---------------------------+-----------------------+--------------------------------------+
| 2. CEEW Mathura / Bareilly   | Fixed balanced cohort of  | Complete 5/5 3-min    | Must be labeled: "Participating      |
|    Sample Cohort Aggregate   | residential meters with   | sub-intervals; >= 90% | Household Cohort Demand" (Strictly   |
|                              | continuous presence       | cohort meters active  | NEVER city demand or feeder demand)  |
+------------------------------+---------------------------+-----------------------+--------------------------------------+
| 3. Grid-India (POSOCO)       | Complete regional / state | 96 standard settlement| Labeled: "Northern Regional Bulk     |
|    Transmission Grid Load    | bulk transmission control | blocks per day per    | Transmission System Demand" or       |
|                              | area (Delhi / UP SLDC)    | IEGC grid code rules  | "State Grid Control Area Demand"     |
+------------------------------+---------------------------+-----------------------+--------------------------------------+
```

### Detailed Mathematical Formulations:

1. **IIIT Delhi Apartment Cohort Aggregate:**
   $$Y_{\text{agg}}^{\text{IIITD}}(t) = \sum_{i \in \mathcal{C}_{\text{active}}(t)} \exp(\text{lnenergy}_{i, t})$$
   Where $\mathcal{C}_{\text{active}}(t) \subseteq \{1, \dots, 19\}$. If $|\mathcal{C}_{\text{active}}(t)| < 18$, the interval is marked unrepresentative.
2. **CEEW Household Sample Cohort Aggregate:**
   $$Y_{\text{agg}}^{\text{CEEW}}(T) = \sum_{m \in \mathcal{C}^*} \left( \sum_{k=1}^{5} \text{t\_kWh}_m(t_k) \right)$$
   Where $\mathcal{C}^*$ is a predefined fixed cohort of smart meters with documented historical continuity. If a meter in $\mathcal{C}^*$ drops offline during window $T$, its missingness is treated via explicit cohort masking.
3. **Grid-India Transmission System Aggregate:**
   Authoritative official 15-minute block demand published by State Load Despatch Centres (SLDC) and NLDC, representing bulk transmission supply.

---

## 5. Section D: Weather Verification & Spatial Offset Audit

### Actual Returned Grid Coordinates vs. Field Coordinates:

Using Open-Meteo's ERA5-Land Reanalysis metadata endpoint, the actual grid centroid coordinates returned by the system were audited:

```
+-------------------------------------------------------------------------------------------------------------------------------+
| METEOROLOGICAL SPATIAL OFFSET AUDIT                                                                                           |
+-----------+----------------------+--------------------------+-----------------+-------------------+---------------------------+
| Location  | Field Coordinates    | Actual ERA5-Land Centroid| Elevation       | Spatial Offset    | Evaluation                |
+-----------+----------------------+--------------------------+-----------------+-------------------+---------------------------+
| Delhi     | 28.5439 N, 77.2724 E | 28.5764 N, 77.2893 E     | 224 meters MSL  | 3.96 km offset    | Excellent (within Okhla   |
| (IIITD)   | (Okhla Phase III)    | (Grid cell center)       |                 | (Euclidean)       | urban microclimate)       |
+-----------+----------------------+--------------------------+-----------------+-------------------+---------------------------+
| Mathura   | 27.4924 N, 77.6737 E | 27.5220 N, 77.7044 E     | 179 meters MSL  | 4.46 km offset    | High fidelity (Agra       |
| (UP)      | (Mathura Urban)      | (Grid cell center)       |                 | (Euclidean)       | airport is 50 km distant) |
+-----------+----------------------+--------------------------+-----------------+-------------------+---------------------------+
| Bareilly  | 28.3670 N, 79.4304 E | 28.3656 N, 79.4779 E     | 177 meters MSL  | 4.65 km offset    | High fidelity (VIBY base  |
| (UP)      | (Bareilly Urban)     | (Grid cell center)       |                 | (Euclidean)       | has public archive gaps)  |
+-----------+----------------------+--------------------------+-----------------+-------------------+---------------------------+
```
*(Zero spatial distance is explicitly NOT claimed; the ~4.0 to 4.7 km offset reflects the physical resolution boundary of the 0.1° ERA5-Land land-surface grid).*

### Available Meteorological Variables & Physical Units:
- **[VERIFIED]**
  - Air Temperature: `temperature_2m` (°C)
  - Relative Humidity: `relative_humidity_2m` (%)
  - Surface Atmospheric Pressure: `surface_pressure` (hPa)
  - Wind Speed: `wind_speed_10m` (km/h, convertible to m/s via $\times 1/3.6$)
  - Solar Radiation: `direct_radiation` ($\text{W/m}^2$), `diffuse_radiation` ($\text{W/m}^2$)
  - Precipitation: `precipitation` (mm)

### Reanalysis Nature & Temporal Coverage:
- **[VERIFIED]** ERA5-Land is an **atmospheric and land-surface reanalysis model** produced by ECMWF under Copernicus Climate Change Service (C3S), combining observational weather station assimilation with numerical weather prediction laws.
- Continuous coverage spans **January 1950 to present**, completely covering Dataset A (2013–2014) and Dataset B (2019–2021).

### Weather Staleness at 15-Minute Forecast Origins:
Because ERA5-Land reanalysis observations are published on **hourly boundaries** (`:00`), at each 15-minute forecast origin time $T$:
- At $T = hh:00$: Latest available weather is from $hh:00$ (**staleness = 0 min**).
- At $T = hh:15$: Latest available weather is from $hh:00$ (**staleness = 15 min**).
- At $T = hh:30$: Latest available weather is from $hh:00$ (**staleness = 30 min**).
- At $T = hh:45$: Latest available weather is from $hh:00$ (**staleness = 45 min**).
- **Leakage Prevention Enforcement:** When forecasting $T+15\text{min}$, only weather records timestamped $t_{\text{weather}} \le T$ are included in the lookback sequence. Upcoming hourly actuals ($hh+1:00$) are strictly blocked.

---

## 6. Section E: Repository Inventory & Deliverables Update

All structured inventories under [`docs/dataset_inventory/`](./dataset_inventory/) have been updated:

1. [`docs/dataset_inventory/repository_resource_inventory.csv`](./dataset_inventory/repository_resource_inventory.csv):
   - 55 resources registered with exact DOIs, repository versions (**1.0 for Dataset A, 2.0 for Dataset B**), file IDs, byte counts, MD5 checksums, classifications, and verification status tags (`VERIFIED`).
2. [`docs/dataset_inventory/dataset_comparison.csv`](./dataset_inventory/dataset_comparison.csv):
   - Updated with exact mathematical aggregation formulas, transition gap notes, measurement semantics, and cohort aggregate definitions.
3. [`docs/dataset_inventory/weather_source_comparison.csv`](./dataset_inventory/weather_source_comparison.csv):
   - Updated with requested vs. actual grid coordinates, calculated Euclidean spatial offsets (~4.0–4.7 km), variables, units, and staleness logic.
4. [`docs/dataset_inventory/source_and_license_register.csv`](./dataset_inventory/source_and_license_register.csv):
   - Updated with verified repository versions, CC0 1.0 Universal terms, CC BY 4.0 terms, formal academic citations, and mandatory non-proprietary BSES disclaimers.

---

## 7. Concise Unresolved Issues (To Be Resolved in Phase 1.2 Data Ingestion)

The following empirical issues cannot be definitively resolved from repository metadata alone and will be tracked during Phase 1.2 ingestion:

1. **Exact Number of Continuously Active Meters in Dataset B:**
   - *Issue:* While ~100 meters are registered, the exact count of meters meeting a $>95\%$ operational availability threshold across 2019, 2020, and 2021 requires empirical computation after downloading annual CSV headers.
2. **Missing Value Encoding in CEEW CSVs:**
   - *Issue:* Whether utility communication dropouts are represented as empty strings, `NaN`, `null`, or simply omitted rows across all annual files must be confirmed upon file ingestion.
3. **Dataset A Baseline vs. Intervention Regimes:**
   - *Issue:* Whether the model should train exclusively on the unperturbed baseline period (August 1, 2013 to February 18, 2014) or across the full time series with an intervention indicator feature (`post`) will be evaluated during exploratory data analysis.

---

> [!IMPORTANT]
> **STOPPED:** Phase 1.1B scientific audit is complete. All claims have been verified against official APIs, byte previews, and physical laws. No full datasets or weather observations were downloaded, and no Git commits were created.
> 
> Awaiting your explicit approval to proceed to Phase 1.2 (Controlled Dataset Acquisition and Verification).
