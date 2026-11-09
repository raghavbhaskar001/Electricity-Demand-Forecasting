# Random Input and Live Inference Verification Report

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala  
**Audit Date:** 2026-11-09  

---

## 1. Executive Summary

The Interactive Prediction Simulator (`src/simulator.py`, `app/serve_dashboard.py`, `app/build_simulator_bundle.py`, and `app/dashboard/simulator.js`) was audited and tested end-to-end across all 36 planned experimental configurations.

| Status Category | Count | Scope & Details |
| :--- | :---: | :--- |
| **Verified Working** | **18 / 36** | All 18 Aggregate Feeder models (`mathura` and `bareilly` × `1D_CNN`, `LSTM`, `CNN_LSTM` × `consumption_only`, `consumption_weather_retro`, `consumption_weather_operational`). Reconciled against stored test predictions with $\max |\Delta| = 8.58 \times 10^{-6}\text{ kW}$ in PyTorch and $5.10 \times 10^{-5}\text{ kW}$ in browser Float32 execution. |
| **Partially Working** | **0 / 36** | None. The interrupted epoch-3 local checkpoint `ind_mathura_1D_CNN_consumption_only.pt` (`val_loss = 0.1244` vs finalized epoch-11 `0.1214`) is explicitly blocked rather than exposed as partially working. |
| **Not Available** | **18 / 36** | All 18 Individual Household Pooled models (`ind_*`). Blocked with explicit diagnostic reason because finalized `.pt` checkpoints and per-meter scalers are pending export from Google Colab (Phase 3.3). Aggregate models are never substituted for individual inference. |
| **Not Tested** | **0 / 36** | All 36 combinations were explicitly tested (18 verified for numerical reconciliation + 18 verified for deterministic rejection). |

---

## 2. Forecasting & Pipeline Specification Audit

| Parameter | Audited Implementation | Verification Status |
| :--- | :--- | :--- |
| **Forecast Horizon ($H$)** | `1` interval (`15 minutes` ahead, $t+1$) | **Verified** (`LOOKBACK_STEPS = 16`, `HORIZON_STEPS = 1`, `INTERVAL_MINUTES = 15`) |
| **Lookback Window ($L$)** | `16` consecutive 15-minute intervals (`4.0 hours`, $[t-15, \dots, t_0]$) | **Verified** |
| **Input & Output Units** | Physical average power demand in **kW** ($P = 4 \times E_{15\text{min}}$) | **Verified** |
| **Feature Ordering** | Strictly matches `src/dataset.py` `FEATURE_SETS`: index `0` is `demand_target`; indices `1..9` are `temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `surface_pressure`, `wind_speed_10m`, `direct_radiation`, `diffuse_radiation`, `cloud_cover`, `precipitation` (`retro_` or `operational_` prefix). | **Verified** |
| **Scaler Fitting** | `StandardScaler` (`feat_scaler`, `targ_scaler`) fitted strictly on the 70% chronological training split (`6,303` windows in Mathura; `6,009` in Bareilly) after 16-step boundary purge. | **Verified** |
| **Evaluation Mode** | `model.eval()` with `torch.no_grad()` (disables `Dropout(0.2)` and freezes `BatchNorm1d` running statistics). | **Verified** |

---

## 3. Complete 36-Combination Inference Availability Matrix

| Experiment ID | Scale | District | Architecture | Feature Config | In Features | Checkpoint File | SHA-256 (16-char) | Best Epoch | Inference Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| `agg_mathura_1D_CNN_consumption_only` | Aggregate | Mathura | 1D CNN | `consumption_only` | 1 | `agg_mathura_1D_CNN_consumption_only.pt` | Verified | 8 | **Verified Working** |
| `agg_mathura_1D_CNN_consumption_weather_retro` | Aggregate | Mathura | 1D CNN | `weather_retro` | 10 | `agg_mathura_1D_CNN_consumption_weather_retro.pt` | Verified | 22 | **Verified Working** |
| `agg_mathura_1D_CNN_consumption_weather_operational` | Aggregate | Mathura | 1D CNN | `weather_operational` | 10 | `agg_mathura_1D_CNN_consumption_weather_operational.pt` | Verified | 12 | **Verified Working** |
| `agg_mathura_LSTM_consumption_only` | Aggregate | Mathura | LSTM | `consumption_only` | 1 | `agg_mathura_LSTM_consumption_only.pt` | Verified | 15 | **Verified Working** |
| `agg_mathura_LSTM_consumption_weather_retro` | Aggregate | Mathura | LSTM | `weather_retro` | 10 | `agg_mathura_LSTM_consumption_weather_retro.pt` | Verified | 7 | **Verified Working** |
| `agg_mathura_LSTM_consumption_weather_operational` | Aggregate | Mathura | LSTM | `weather_operational` | 10 | `agg_mathura_LSTM_consumption_weather_operational.pt` | Verified | 11 | **Verified Working** |
| `agg_mathura_CNN_LSTM_consumption_only` | Aggregate | Mathura | CNN–LSTM | `consumption_only` | 1 | `agg_mathura_CNN_LSTM_consumption_only.pt` | Verified | 20 | **Verified Working** |
| `agg_mathura_CNN_LSTM_consumption_weather_retro` | Aggregate | Mathura | CNN–LSTM | `weather_retro` | 10 | `agg_mathura_CNN_LSTM_consumption_weather_retro.pt` | Verified | 11 | **Verified Working** |
| `agg_mathura_CNN_LSTM_consumption_weather_operational` | Aggregate | Mathura | CNN–LSTM | `weather_operational` | 10 | `agg_mathura_CNN_LSTM_consumption_weather_operational.pt` | Verified | 11 | **Verified Working** |
| `agg_bareilly_1D_CNN_consumption_only` | Aggregate | Bareilly | 1D CNN | `consumption_only` | 1 | `agg_bareilly_1D_CNN_consumption_only.pt` | Verified | 14 | **Verified Working** |
| `agg_bareilly_1D_CNN_consumption_weather_retro` | Aggregate | Bareilly | 1D CNN | `weather_retro` | 10 | `agg_bareilly_1D_CNN_consumption_weather_retro.pt` | Verified | 20 | **Verified Working** |
| `agg_bareilly_1D_CNN_consumption_weather_operational` | Aggregate | Bareilly | 1D CNN | `weather_operational` | 10 | `agg_bareilly_1D_CNN_consumption_weather_operational.pt` | Verified | 8 | **Verified Working** |
| `agg_bareilly_LSTM_consumption_only` | Aggregate | Bareilly | LSTM | `consumption_only` | 1 | `agg_bareilly_LSTM_consumption_only.pt` | Verified | 12 | **Verified Working** |
| `agg_bareilly_LSTM_consumption_weather_retro` | Aggregate | Bareilly | LSTM | `weather_retro` | 10 | `agg_bareilly_LSTM_consumption_weather_retro.pt` | Verified | 5 | **Verified Working** |
| `agg_bareilly_LSTM_consumption_weather_operational` | Aggregate | Bareilly | LSTM | `weather_operational` | 10 | `agg_bareilly_LSTM_consumption_weather_operational.pt` | Verified | 5 | **Verified Working** |
| `agg_bareilly_CNN_LSTM_consumption_only` | Aggregate | Bareilly | CNN–LSTM | `consumption_only` | 1 | `agg_bareilly_CNN_LSTM_consumption_only.pt` | Verified | 24 | **Verified Working** |
| `agg_bareilly_CNN_LSTM_consumption_weather_retro` | Aggregate | Bareilly | CNN–LSTM | `weather_retro` | 10 | `agg_bareilly_CNN_LSTM_consumption_weather_retro.pt` | Verified | 14 | **Verified Working** |
| `agg_bareilly_CNN_LSTM_consumption_weather_operational` | Aggregate | Bareilly | CNN–LSTM | `weather_operational` | 10 | `agg_bareilly_CNN_LSTM_consumption_weather_operational.pt` | Verified | 14 | **Verified Working** |
| `ind_mathura_1D_CNN_consumption_only` | Individual | Mathura | 1D CNN | `consumption_only` | 1 | `ind_mathura_1D_CNN_consumption_only.pt` (Partial) | Excluded | 3 (vs 11) | **Not Available** (Partial run excluded) |
| Remaining 17 `ind_*` combinations | Individual | Mathura / Bareilly | All 3 | All 3 | 1 / 10 | None locally | — | — | **Not Available** (Pending Colab export) |

---

## 4. Verification of Input Modes & Safeguards

### 4.1 Manual Input Verification
- Accepts 16 chronologically ordered consumption values ($t-15$ through $t_0$) via 16 individual interval fields or a quick-paste parser.
- Rejects sequences with fewer or more than 16 values, missing/empty entries, `NaN`/`Inf`, boolean values, non-numeric strings, and negative kW demand (`< 0.0 kW`).
- Checks every observation against the selected district's 70% training distribution:
  - **Mathura Aggregate Training Bounds:** $\min = 1.168\text{ kW}$, $p_{01} = 8.204\text{ kW}$, $p_{05} = 10.128\text{ kW}$, $p_{50} = 18.004\text{ kW}$, $p_{95} = 34.352\text{ kW}$, $p_{99} = 36.864\text{ kW}$, $\max = 39.060\text{ kW}$.
  - **Bareilly Aggregate Training Bounds:** $\min = 1.648\text{ kW}$, $p_{01} = 3.608\text{ kW}$, $p_{05} = 4.708\text{ kW}$, $p_{50} = 9.672\text{ kW}$, $p_{95} = 14.800\text{ kW}$, $p_{99} = 17.048\text{ kW}$, $\max = 23.344\text{ kW}$.
- Emits `consumption_outside_training_range` (outside $[\min, \max]$) or `consumption_in_extreme_tail` (outside $[p_{01}, p_{99}]$) warnings **without silently clipping** user inputs (`clipped = False` verified in unit tests).
- For weather-dependent configurations (`consumption_weather_retro`, `consumption_weather_operational`), never fabricates weather: requires a $16 \times 9$ meteorological matrix (loaded from a real pipeline historical window, paired training scenario window, uploaded 10-column CSV, or manual $16 \times 9$ editor).

### 4.2 Statistically Grounded Random Scenario Generator
- All 6 presets (`Normal Consumption`, `High Demand`, `Low Demand`, `Morning Peak`, `Evening Peak`, `Irregular Consumption`) sample real 16-step trajectories from the selected district's 70% training split (`6,303` Mathura windows; `6,009` Bareilly windows):
  - `normal`: $p_{35} \le \mu_w \le p_{65}$ and mean $|\Delta x| \le p_{75}$ (`1,173` Mathura / `1,193` Bareilly windows).
  - `high_demand`: $\mu_w \ge p_{90}$ (`631` Mathura / `601` Bareilly windows).
  - `low_demand`: $\mu_w \le p_{10}$ (`631` Mathura / `601` Bareilly windows).
  - `morning_peak`: window ending $06:30\text{–}10:45\text{ IST}$ with peak above morning median (`255` Mathura / `432` Bareilly windows).
  - `evening_peak`: window ending $18:30\text{–}22:45\text{ IST}$ with peak above evening median and positive ramp (`509` Mathura / `460` Bareilly windows).
  - `irregular`: mean 15-minute step change $|\Delta x| \ge p_{90}$ (`631` Mathura / `601` Bareilly windows).
- Adds a small covariance-preserving innovation $\delta = 0.12 L_{\text{Chol}} z$ ($z \sim \mathcal{N}(0, I_{16})$) computed from the regime's empirical $16 \times 16$ lag covariance matrix using a reproducible random seed (`np.random.default_rng(seed)` in Python; seeded `mulberry32` + Box–Muller in JS).
- **UI Distinction:** The simulator interface explicitly notes that predictions on randomly generated or manually entered trajectories are exploratory model response simulations, not evaluations of real-world forecasting accuracy.

### 4.3 CSV Upload Verification
- Supports consumption-only CSVs (`demand_target`, `consumption_kw`, `aggregate_demand_kw_primary`, `actual_demand_kw`) and 10-column weather-augmented CSVs (`demand_target` + 9 exact weather columns).
- Supports selecting any contiguous 16-row slice via `row_offset` when the uploaded CSV has $> 16$ rows.
- Explicitly validates and rejects with descriptive error messages:
  - Empty files or $< 16$ data rows.
  - Missing or non-numeric cell values.
  - Duplicate timestamps (`CSV timestamp column contains duplicate timestamps`).
  - Non-consecutive sampling intervals (`\Delta t \ne 15\text{ min}`).
  - Invalid/unparseable timestamp strings.
  - Missing required consumption or weather column names.
  - Physically impossible values (negative kW demand, relative humidity or cloud cover outside $[0, 100]\%$, negative wind speed/radiation/precipitation, pressure outside $(500, 1150)\text{ hPa}$).

---

## 5. Executed Verification Tests Summary

| Test Case | Test Location | Result |
| :--- | :--- | :--- |
| 36-combination availability matrix & partial checkpoint exclusion | `tests/test_simulator.py::test_availability_matrix_completeness_and_integrity` | **PASS** |
| Blocking of unsupported scale (`individual`) & unknown district (`delhi`) | `tests/test_simulator.py::test_unsupported_combinations_blocked` | **PASS** |
| Reconciliation against `aggregate_test_predictions.parquet` across all 18 models (54 test windows in Python; 72 in JS Float32 engine) | `tests/test_simulator.py::test_exact_reconciliation_all_18_aggregate_models` & Node E2E harness | **PASS** ($\max |\Delta| = 8.58 \times 10^{-6}\text{ kW}$ PyTorch; $5.10 \times 10^{-5}\text{ kW}$ JS) |
| Manual input rejection of `<16`, `>16`, `None`, `""`, `"abc"`, `NaN`, `Inf`, `-0.5 kW` | `tests/test_simulator.py::test_manual_input_validation_rejects_invalid_sequences` | **PASS** |
| Non-fabrication of weather features on `retro` and `operational` models | `tests/test_simulator.py::test_weather_configurations_never_fabricate_weather` | **PASS** |
| OOD warning emission on extreme values (`92.5 kW`) without clipping (`clipped=False`) | `tests/test_simulator.py::test_ood_warning_without_silent_clipping` | **PASS** |
| Seeded reproducibility, regime ordering (`low < normal < high`), & paired weather across all 6 scenarios | `tests/test_simulator.py::test_random_scenario_generator_reproducibility_and_regimes` | **PASS** |
| CSV parsing (consumption-only, 10-col weather, `row_offset`, non-15-min gap, duplicate timestamps, invalid timestamp) | `tests/test_simulator.py::test_csv_input_consumption_and_weather` | **PASS** |
