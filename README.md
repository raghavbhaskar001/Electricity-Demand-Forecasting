# Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models

**Author:** Raghav Bhaskar  
**Department:** Department of Electrical Engineering  
**Institution:** Thapar Institute of Engineering & Technology, Patiala

---

## 1. Overview

This repository contains the data pipeline, deep learning architectures, statistical evaluation framework, and interactive simulation interface for a comparative study of short-term electricity demand forecasting on Indian residential distribution grids. The study evaluates three neural sequence architectures—**1D Convolutional Neural Networks (`1D_CNN`)**, **Long Short-Term Memory networks (`LSTM`)**, and **Hybrid `CNN_LSTM` networks**—against deterministic persistence baselines across **36 controlled experiments** spanning two geographic districts in Uttar Pradesh (**Mathura** and **Bareilly**), two spatial aggregation scales (**Aggregate Feeder Scale** and **Individual Household Meter Scale**), and three meteorological covariate regimes.

All models forecast next-15-minute average active power ($\hat{y}_{t+1}$ in physical $\text{kW}$, horizon $H = 1$) from a contiguous 16-step ($4.0\text{-hour}$) historical lookback window using leak-free chronological partitioning ($70\% / 15\% / 15\%$) with explicit $L = 16$ boundary purge gaps.

---

## 2. Research Motivation & Engineering Context

Distribution utilities and microgrid operators rely on 15-minute load forecasts for feeder dispatch, peak-load management, and imbalance settlement. While deep neural networks are widely proposed for short-term load forecasting, two practical questions often remain unresolved in the literature:

1. **Spatial Aggregation vs. Model Complexity:** Does deep learning add predictive skill at the smoothed aggregate feeder level ($N = 38\text{–}46$ residences) where autocorrelation is high but training sample counts are modest (~6,000 windows), or does its advantage emerge only at the volatile individual household meter level (~1.1M–1.9M windows)?
2. **Meteorological Covariate Realism:** Many studies pair historical load with contemporaneous reanalysis weather (`retrospective`), ignoring the 5-day publication latency of verified reanalysis products such as ERA5-Land (`operational`). How do retrospective and 120-hour lagged weather inputs affect out-of-sample generalization relative to univariate load history (`consumption_only`)?

---

## 3. Research Questions

1. **RQ1 (Architecture Comparison):** How do `1D_CNN`, `LSTM`, and `CNN_LSTM` compare in 15-minute ahead forecasting accuracy across aggregate feeder series and pooled individual smart-meter panels?
2. **RQ2 (Scale Dependency):** Under identical preprocessing and optimization, how does deep learning skill relative to Naive Persistence ($\hat{y}_{t+1} = y_t$) invert between the aggregate feeder scale and the individual household scale?
3. **RQ3 (Exogenous Weather Utility):** Do contemporaneous (`consumption_weather_retro`) or 120-hour operationally lagged (`consumption_weather_operational`) ERA5-Land meteorological covariates improve 15-minute forecasting performance over `consumption_only`?
4. **RQ4 (Loss Function Trade-offs):** How does Huber loss optimization (`delta = 1.0`) trade off squared-error spike suppression (`RMSE`) against median/flat-regime absolute error (`MAE`) on volatile household meters?

---

## 4. Key Empirical Findings

1. **Scale Inversion Across Spatial Tiers:**
   - **Aggregate Feeder Scale (`18 / 18` experiments):** The zero-parameter **Naive Persistence baseline** ($\hat{y}_{t+1} = y_t$) outperforms all 18 deep learning models in both Mathura (`RMSE = 1.2033 kW` vs. best neural `1.5298 kW`, Diebold–Mariano $p_{\text{adj}} < 10^{-15}$) and Bareilly (`RMSE = 2.3063 kW` vs. best neural `2.9540 kW`, $p_{\text{adj}} = 7.47 \times 10^{-10}$). Aggregation smooths stochastic appliance switching, yielding high lag-1 autocorrelation ($\rho_1 > 0.93$) where ~6,000 training windows are insufficient for `52K–71K`-parameter networks to surpass the identity mapping.
   - **Individual Household Scale (`18 / 18` experiments):** Across `229,281` test sequences in Mathura and `398,897` test sequences in Bareilly, **all 18 deep learning models outperform Naive Persistence on `RMSE`, $R^2$, and `Peak_MAE`**. In Mathura, `CNN_LSTM` (`consumption_weather_operational`) achieves `RMSE = 0.1880 kW` (`+20.57%` skill over persistence `0.2110 kW`, $R^2 = 0.6826$ vs. `0.6002`). In Bareilly, `LSTM` (`consumption_weather_operational`) achieves `RMSE = 0.2548 kW` (`+14.40%` skill over persistence `0.2755 kW`, $R^2 = 0.7796$ vs. `0.7425`).
2. **Exogenous Weather Overfitting on Small Samples:**
   - On aggregate feeder series, adding 9 ERA5-Land weather features increases out-of-sample `RMSE` by `23%–64%` in Mathura and `45%–51%` in Bareilly compared to `consumption_only`. Hourly-to-15-minute interpolated regional reanalysis adds collinearity without 15-minute predictive signal when training sample counts are small.
   - On individual household panels ($1.08\text{M}–1.87\text{M}$ training windows), models have sufficient sample support to regularize the 10-channel input space without overfitting.
3. **`RMSE` vs. `MAE` Divergence under Huber Loss:**
   - While all 9 Bareilly individual models beat Naive Persistence on `RMSE` (`0.2548–0.2591 kW` vs. `0.2755 kW`) and `Peak_MAE` (`0.4270–0.4408 kW` vs. `0.4832 kW`), none beat Naive Persistence on pooled `MAE` (`0.1132–0.1154 kW` vs. `0.1114 kW`). Huber loss shrinks large step-change errors during appliance switching (lowering `RMSE`) at the cost of a ~`2–4 W` offset during long flat standby intervals where $\hat{y}_{t+1} = y_t$ is nearly exact.

---

## 5. Datasets & Study Regions

| Cohort | Upstream Source | Coordinates | Native Cadence | Active Cohort ($\ge 90\%$ Coverage) | Study Period | Modelling Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mathura** (Uttar Pradesh) | CEEW Smart Meter Dataset (`doi:10.7910/DVN/N3615P`) | `27.4924° N, 77.6737° E` | 3 min | **38 meters** | May 2019 – May 2020 (`35,127` 15-min steps) | Primary Aggregate & Individual Modelling |
| **Bareilly** (Uttar Pradesh) | CEEW Smart Meter Dataset (`doi:10.7910/DVN/N3615P`) | `28.3670° N, 79.4304° E` | 3 min | **46 meters** | May 2019 – Oct 2020 (`49,336` 15-min steps) | Primary Aggregate & Individual Modelling |
| **New Delhi** | IIIT Delhi Residential Dataset (`doi:10.6084/m9.figshare.c.4490387`) | `28.5450° N, 77.2732° E` | 1 min | Exploratory | `370` 15-min steps (`96` windows; `0` test windows) | Exploratory Descriptive Analysis Only |
| **ERA5-Land Weather** | Copernicus C3S via Open-Meteo (`doi:10.24381/cds.e2161bac`) | District centroids | 60 min | 9 meteorological channels | Matched to load spans | Exogenous Weather Covariates |

---

## 6. Data Processing & Quality Control Pipeline

Implemented in `scripts/process_smart_meter.py`, `scripts/align_weather_and_load.py`, and `src/dataset.py`:
1. **Exact Sub-Interval Completeness (`5 x 3-min = 15-min`):** A 15-minute interval is retained only if all 5 constituent 3-minute readings exist (`valid_obs_count == 5`). Mean active power in $\text{kW}$ is computed as $P_{\text{kW}} = \frac{1}{5}\sum_{j=1}^{5} (E_{3\text{min},j} \times 20)$.
2. **Fixed Cohort Stability ($\ge 90\%$ Completeness):** Aggregate feeder load sums only meters with $\ge 90\%$ lifetime coverage (`38` in Mathura, `46` in Bareilly), and a 15-minute aggregate timestamp is marked clean (`is_clean_agg_primary == True`) only when $\ge 90\%$ of cohort meters report valid 5-of-5 readings simultaneously.
3. **Strict 17-Step Window Contiguity:** Sliding windows ($L = 16$ inputs $+ H = 1$ target $= 17$ timestamps) are extracted strictly from unbroken 15-minute sequences ($\Delta t = 15\text{ min}$ at every step). No imputation is performed across telemetry dropouts.

---

## 7. Meteorological Integration & Operational Lag Regimes

Nine ERA5-Land meteorological variables (`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `surface_pressure`, `wind_speed_10m`, `direct_radiation`, `diffuse_radiation`, `cloud_cover`, `precipitation`) are aligned to 15-minute intervals under two regimes:
- **Retrospective (`consumption_weather_retro`):** Contemporaneous ERA5-Land reanalysis at timestamp $t$ (`0h` lag), representing an idealized hindcast.
- **Operational (`consumption_weather_operational`):** Enforces a **120-hour (5-day) availability lag** (`t - 120h`), reflecting the real-world publication delay of verified ERA5-Land reanalysis when live local meteorological sensors are unavailable.

---

## 8. Feature Configurations

| Feature Configuration Name | Input Channels ($C$) | Feature List |
| :--- | :---: | :--- |
| **`consumption_only`** | `1` | `demand_target` ($\text{kW}$) |
| **`consumption_weather_retro`** | `10` | `demand_target` + 9 `retro_*` ERA5-Land variables |
| **`consumption_weather_operational`** | `10` | `demand_target` + 9 `operational_*` (120h-lagged) ERA5-Land variables |

---

## 9. Model Architectures & Parameter Counts

Implemented in `src/models.py` (`seq_len = 16`):

| Architecture | PyTorch Class | Layer Structure | Params ($C=1$) | Params ($C=10$) |
| :--- | :--- | :--- | ---: | ---: |
| **`1D_CNN`** | `CNN1DForecaster` | `2 x [Conv1d(k=3, pad=1, ch=64) -> BatchNorm1d -> GELU -> Dropout(0.2)] -> Flatten(1024) -> Linear(1024, 32) -> GELU -> Linear(32, 1)` | **52,225** | **53,953** |
| **`LSTM`** | `LSTMForecaster` | `LSTM(hidden=64, layers=2, dropout=0.2) -> last_step(64) -> Dropout(0.2) -> Linear(64, 32) -> GELU -> Linear(32, 1)` | **52,545** | **54,849** |
| **`CNN_LSTM`** | `CNNLSTMForecaster` | `2 x [Conv1d(k=3, pad=1, ch=64) -> BatchNorm1d -> GELU] -> LSTM(hidden=64, layers=2, dropout=0.2) -> Dropout(0.2) -> Linear(64, 32) -> GELU -> Linear(32, 1)` | **69,057** | **70,785** |

---

## 10. Baseline Models

Implemented in `src/baselines.py` and evaluated on the identical locked test sequences (`artifacts/metrics/baseline_benchmarks.json`):
1. **Naive Persistence (`naive_persistence`):** $\hat{y}_{t+1} = y_t$ (carries forward the most recent 15-minute demand reading).
2. **Seasonal Previous-Day Persistence (`seasonal_persistence_24h`):** $\hat{y}_{t+1} = y_{t+1-96}$ (same 15-minute time slot 24 hours earlier, evaluated on the subset of test windows with an unbroken 96-step lookback).
3. **Moving Average (`moving_average_16`):** Mean of the 16-step (4-hour) lookback window: $\hat{y}_{t+1} = \frac{1}{16}\sum_{k=0}^{15} y_{t-k}$.

---

## 11. Training Configuration & Leak-Free Split Protocol

| Hyperparameter / Protocol | Aggregate Feeder Experiments (`18` runs) | Individual Household Experiments (`18` runs) |
| :--- | :--- | :--- |
| **Chronological Split** | `70% Train / 15% Val / 15% Test` | `70% Train / 15% Val / 15% Test` (applied per `meter_id`) |
| **Boundary Purge Gap** | `L = 16` windows purged at `Train->Val` and `Val->Test` | `L = 16` windows purged per meter (`1,216` Mathura; `1,472` Bareilly) |
| **Sequence Census (`Train / Val / Test`)** | Mathura: `6,303 / 1,350 / 1,320`<br>Bareilly: `6,009 / 1,287 / 1,257` | Mathura: `1,075,464 / 230,441 / 229,281`<br>Bareilly: `1,868,167 / 400,306 / 398,897` |
| **Feature & Target Normalization** | Global `StandardScaler` fit strictly on `Train` timestamps | Per-meter `StandardScaler` fit strictly on each meter's `Train` timestamps |
| **Loss Function** | `nn.HuberLoss(delta=1.0)` | `nn.HuberLoss(delta=1.0)` |
| **Optimizer & Weight Decay** | `AdamW(lr=1e-3, weight_decay=1e-4)` | `AdamW(lr=1e-3, weight_decay=1e-4)` |
| **Gradient Clipping** | `clip_grad_norm_(max_norm=1.0)` | `clip_grad_norm_(max_norm=1.0)` |
| **LR Scheduler** | `ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)` | `ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)` |
| **Batch Size / Max Epochs / Patience** | `batch_size = 64`, `epochs = 40`, `patience = 8` | `batch_size = 512`, `epochs = 15`, `patience = 6` |

---

## 12. Evaluation Metrics

All metrics (`src/evaluation.py`) are computed in physical $\text{kW}$ after inverse-transforming predictions via `scaler_y`:
- **RMSE ($\text{kW}$):** Root Mean Squared Error (primary quadratic error metric).
- **MAE ($\text{kW}$):** Mean Absolute Error.
- **WAPE ($\%$):** Weighted Absolute Percentage Error, $100 \times \sum |y_i - \hat{y}_i| / \sum |y_i|$.
- **MAPE ($\%$):** Mean Absolute Percentage Error filtering near-zero intervals ($y_i \ge 0.01\text{ kW}$).
- **$R^2$:** Coefficient of determination.
- **nRMSE ($\%$):** RMSE normalized by empirical test mean load.
- **Persistence Skill Score ($\%$):** $100 \times (1 - \text{MSE}_{\text{model}} / \text{MSE}_{\text{persistence}})$.
- **Peak Load MAE ($\text{kW}$):** MAE on the top $5\%$ highest-demand test intervals ($y_i \ge P_{95}$).

---

## 13. Experimental Results — Aggregate Feeder Scale (`18 / 18` Verified)

All 18 aggregate feeder models have verified `.pt` checkpoints (`artifacts/checkpoints/agg_*.pt`), pointwise test predictions (`artifacts/metrics/aggregate_predictions.parquet`), and Diebold–Mariano tests (`artifacts/metrics/dm_statistical_tests.json`).

### 13.1 Mathura Aggregate Feeder (`38` Meters, `1,320` Test Sequences)

| Rank | Model / Baseline | Feature Configuration | Best / Total Epochs | RMSE (kW) | MAE (kW) | WAPE (%) | $R^2$ | Peak MAE (kW) | Skill vs. Persistence (%) |
| :---: | :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline** | **Naive Persistence** ($\hat{y}_{t+1}=y_t$) | `consumption_only` | — | **1.2033** | **0.8713** | **11.22%** | **0.8825** | **1.5469** | **0.00%** |
| Baseline | Moving Average ($L=16$) | `consumption_only` | — | 2.0947 | 1.6305 | 21.00% | 0.6439 | 3.7484 | -202.98% |
| Baseline | Seasonal Persistence ($24\text{h}$, $N=749$) | `consumption_only` | — | 2.0970 | 1.5471 | 19.65% | 0.6426 | 2.7110 | -205.51% |
| 1 | `LSTM` | `consumption_only` | 15 / 23 | 1.5298 | 1.2797 | 16.48% | 0.8101 | 2.1385 | -61.64% |
| 2 | `1D_CNN` | `consumption_only` | 8 / 16 | 1.5490 | 1.2885 | 16.60% | 0.8054 | 1.9977 | -65.71% |
| 3 | `CNN_LSTM` | `consumption_only` | 7 / 15 | 1.6820 | 1.4090 | 18.15% | 0.7705 | 2.5594 | -95.39% |
| 4 | `LSTM` | `consumption_weather_retro` | 4 / 12 | 1.8819 | 1.5005 | 19.33% | 0.7127 | 2.8639 | -144.60% |
| 5 | `CNN_LSTM` | `consumption_weather_retro` | 2 / 10 | 1.9765 | 1.6085 | 20.72% | 0.6831 | 3.0439 | -169.80% |
| 6 | `1D_CNN` | `consumption_weather_retro` | 7 / 15 | 2.0631 | 1.6815 | 21.66% | 0.6547 | 2.7554 | -193.96% |
| 7 | `LSTM` | `consumption_weather_operational` | 3 / 11 | 2.5015 | 1.9810 | 25.52% | 0.4924 | 3.1656 | -332.16% |
| 8 | `1D_CNN` | `consumption_weather_operational` | 7 / 15 | 2.5364 | 2.0545 | 26.47% | 0.4781 | 2.6606 | -344.29% |
| 9 | `CNN_LSTM` | `consumption_weather_operational` | 1 / 9 | 2.7850 | 2.2484 | 28.96% | 0.3708 | 4.8686 | -435.67% |

### 13.2 Bareilly Aggregate Feeder (`46` Meters, `1,257` Test Sequences)

| Rank | Model / Baseline | Feature Configuration | Best / Total Epochs | RMSE (kW) | MAE (kW) | WAPE (%) | $R^2$ | Peak MAE (kW) | Skill vs. Persistence (%) |
| :---: | :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline** | **Naive Persistence** ($\hat{y}_{t+1}=y_t$) | `consumption_only` | — | **2.3063** | **1.7289** | **8.97%** | **0.8452** | **2.8781** | **0.00%** |
| 1 | `1D_CNN` | `consumption_only` | 14 / 22 | 2.9540 | 2.2348 | 11.60% | 0.7461 | 2.8648 | -64.05% |
| Baseline | Seasonal Persistence ($24\text{h}$, $N=718$) | `consumption_only` | — | 3.5508 | 2.6512 | 14.07% | 0.6199 | 3.4384 | -135.58% |
| 2 | `LSTM` | `consumption_only` | 12 / 20 | 3.8015 | 2.7655 | 14.35% | 0.5795 | 3.7493 | -171.69% |
| 3 | `CNN_LSTM` | `consumption_only` | 4 / 12 | 4.2165 | 3.1763 | 16.49% | 0.4827 | 5.2932 | -234.25% |
| 4 | `1D_CNN` | `consumption_weather_retro` | 5 / 13 | 4.2940 | 3.2496 | 16.87% | 0.4635 | 4.5015 | -246.65% |
| Baseline | Moving Average ($L=16$) | `consumption_only` | — | 4.3249 | 3.3143 | 17.20% | 0.4557 | 5.6435 | -251.67% |
| 5 | `1D_CNN` | `consumption_weather_operational` | 3 / 11 | 4.4628 | 3.6054 | 18.71% | 0.4205 | 3.6660 | -274.44% |
| 6 | `CNN_LSTM` | `consumption_weather_retro` | 2 / 10 | 4.5579 | 3.5621 | 18.49% | 0.3955 | 6.6660 | -290.57% |
| 7 | `LSTM` | `consumption_weather_retro` | 3 / 11 | 4.6513 | 3.6352 | 18.87% | 0.3705 | 6.5396 | -306.74% |
| 8 | `CNN_LSTM` | `consumption_weather_operational` | 2 / 10 | 4.7495 | 3.8299 | 19.88% | 0.3436 | 7.1257 | -324.10% |
| 9 | `LSTM` | `consumption_weather_operational` | 3 / 11 | 4.9613 | 4.0067 | 20.80% | 0.2838 | 6.7277 | -362.78% |

---

## 14. Experimental Results — Individual Household Scale (`18 / 18` Trained & Logged)

All 18 individual household models were trained and evaluated on the locked chronological test sets in Google Colab GPU (`artifacts/metrics/training_results.json`), reporting both pooled sequence metrics and per-meter cross-sectional distributions (median and interquartile range across meters). Binary `.pt` checkpoint export from Colab is tracked under Phase 3.3.

### 14.1 Mathura Individual Household Panel (`38` Meters, `229,281` Test Sequences)

| Rank (RMSE) | Model / Baseline | Feature Configuration | Best / Total Epochs | Pooled RMSE (kW) | Pooled MAE (kW) | WAPE (%) | $R^2$ | Peak MAE (kW) | Skill vs. Persistence (%) | Median Meter RMSE (IQR, kW) |
| :---: | :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **`CNN_LSTM`** | `consumption_weather_operational` | 11 / 15 | **0.1880** | 0.0680 | 36.74% | **0.6826** | **0.3087** | **+20.57%** | 0.1311 (0.1456) |
| 2 | **`CNN_LSTM`** | `consumption_weather_retro` | 8 / 14 | 0.1886 | **0.0671** | **36.19%** | 0.6804 | 0.3127 | +20.06% | **0.1305** (0.1426) |
| 3 | `LSTM` | `consumption_only` | 15 / 15 | 0.1888 | 0.0673 | 36.32% | 0.6799 | 0.3100 | +19.91% | 0.1322 (0.1469) |
| 4 | `LSTM` | `consumption_weather_operational` | 10 / 15 | 0.1890 | 0.0684 | 36.93% | 0.6791 | 0.3125 | +19.72% | 0.1311 (0.1449) |
| 5 | `CNN_LSTM` | `consumption_only` | 15 / 15 | 0.1891 | 0.0684 | 36.93% | 0.6789 | 0.3122 | +19.65% | 0.1321 (0.1462) |
| 6 | `LSTM` | `consumption_weather_retro` | 3 / 9 | 0.1904 | 0.0721 | 38.94% | 0.6743 | 0.3239 | +18.51% | 0.1342 (0.1440) |
| 7 | `1D_CNN` | `consumption_only` | 13 / 15 | 0.1912 | 0.0695 | 37.54% | 0.6716 | 0.3177 | +17.85% | 0.1333 (0.1465) |
| 8 | `1D_CNN` | `consumption_weather_operational` | 5 / 11 | 0.1916 | 0.0714 | 38.57% | 0.6702 | 0.3210 | +17.50% | 0.1320 (0.1462) |
| 9 | `1D_CNN` | `consumption_weather_retro` | 11 / 15 | 0.1920 | 0.0703 | 37.94% | 0.6688 | 0.3249 | +17.14% | 0.1344 (0.1444) |
| **Baseline** | **Naive Persistence** ($\hat{y}_{t+1}=y_t$) | `consumption_only` | — | **0.2110** | **0.0707** | **38.13%** | **0.6002** | **0.3536** | **0.00%** | — |
| Baseline | Moving Average ($L=16$) | `consumption_only` | — | 0.2442 | 0.1112 | 60.03% | 0.4642 | 0.4912 | -34.04% | — |

### 14.2 Bareilly Individual Household Panel (`46` Meters, `398,897` Test Sequences)

| Rank (RMSE) | Model / Baseline | Feature Configuration | Best / Total Epochs | Pooled RMSE (kW) | Pooled MAE (kW) | WAPE (%) | $R^2$ | Peak MAE (kW) | Skill vs. Persistence (%) | Median Meter RMSE (IQR, kW) |
| :---: | :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **`LSTM`** | `consumption_weather_operational` | 14 / 15 | **0.2548** | 0.1137 | 27.79% | **0.7796** | 0.4310 | **+14.40%** | 0.1384 (0.1341) |
| 2 | `CNN_LSTM` | `consumption_weather_operational` | 15 / 15 | 0.2557 | 0.1145 | 27.99% | 0.7782 | **0.4270** | +13.84% | **0.1369** (0.1349) |
| 3 | `CNN_LSTM` | `consumption_only` | 15 / 15 | 0.2559 | 0.1140 | 27.86% | 0.7779 | 0.4281 | +13.71% | 0.1388 (0.1376) |
| 4 | `LSTM` | `consumption_only` | 15 / 15 | 0.2563 | 0.1140 | 27.88% | 0.7771 | 0.4292 | +13.43% | 0.1391 (0.1366) |
| 5 | `LSTM` | `consumption_weather_retro` | 15 / 15 | 0.2564 | **0.1132** | **27.69%** | 0.7769 | 0.4337 | +13.34% | 0.1404 (0.1371) |
| 6 | `CNN_LSTM` | `consumption_weather_retro` | 10 / 15 | 0.2571 | 0.1145 | 28.00% | 0.7757 | 0.4342 | +12.86% | 0.1419 (0.1356) |
| 7 | `1D_CNN` | `consumption_weather_operational` | 14 / 15 | 0.2577 | 0.1146 | 28.03% | 0.7746 | 0.4341 | +12.46% | 0.1383 (0.1385) |
| 8 | `1D_CNN` | `consumption_only` | 15 / 15 | 0.2581 | 0.1154 | 28.22% | 0.7739 | 0.4317 | +12.19% | 0.1401 (0.1378) |
| 9 | `1D_CNN` | `consumption_weather_retro` | 14 / 15 | 0.2591 | 0.1153 | 28.19% | 0.7722 | 0.4408 | +11.54% | 0.1419 (0.1395) |
| **Baseline** | **Naive Persistence** ($\hat{y}_{t+1}=y_t$) | `consumption_only` | — | **0.2755** | **0.1114** | **27.22%** | **0.7425** | **0.4832** | **0.00%** | — |
| Baseline | Moving Average ($L=16$) | `consumption_only` | — | 0.3408 | 0.1757 | 42.96% | 0.6061 | 0.6788 | -53.03% | — |

---

## 15. Statistical Significance Analysis (Diebold–Mariano & Holm–Bonferroni)

Pairwise forecast comparisons on the aggregate feeder test sets (`scripts/run_dm_tests.py`, `artifacts/metrics/dm_statistical_tests.csv`) use the Harvey–Leybourne–Newbold (1997) finite-sample corrected Diebold–Mariano statistic with Newey–West HAC variance estimation and Holm–Bonferroni adjustment ($K = 12$ hypotheses per district family):

| District | Hypothesis Category | Comparison (`Model 1` vs. `Model 2`) | DM Statistic ($t_{\text{HLN}}$) | Raw $p$-Value | Holm $p_{\text{adj}}$ | Statistically Significant Winner ($\alpha=0.05$) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| **Mathura** | Best DL vs. Baseline | `LSTM (consumption_only)` vs. `Naive Persistence` | $+9.9722$ | $<10^{-15}$ | $<10^{-15}$ | **Naive Persistence** |
| **Mathura** | Architecture (`consumption_only`) | `1D_CNN` vs. `LSTM` | $+0.5174$ | $0.6050$ | $0.6050$ | Tie ($p_{\text{adj}} = 0.605$) |
| **Mathura** | Architecture (`consumption_only`) | `1D_CNN` vs. `CNN_LSTM` | $-3.9498$ | $8.24 \times 10^{-5}$ | $3.30 \times 10^{-4}$ | **`1D_CNN`** |
| **Mathura** | Architecture (`consumption_only`) | `LSTM` vs. `CNN_LSTM` | $-10.2286$ | $<10^{-15}$ | $<10^{-15}$ | **`LSTM`** |
| **Mathura** | Weather Ablation (`LSTM`) | `consumption_only` vs. `weather_retro` | $-10.7495$ | $<10^{-15}$ | $<10^{-15}$ | **`consumption_only`** |
| **Mathura** | Weather Ablation (`LSTM`) | `consumption_only` vs. `weather_operational` | $-16.2508$ | $<10^{-15}$ | $<10^{-15}$ | **`consumption_only`** |
| **Bareilly** | Best DL vs. Baseline | `1D_CNN (consumption_only)` vs. `Naive Persistence` | $+6.3150$ | $3.74 \times 10^{-10}$ | $7.47 \times 10^{-10}$ | **Naive Persistence** |
| **Bareilly** | Architecture (`consumption_only`) | `1D_CNN` vs. `LSTM` | $-9.7347$ | $<10^{-15}$ | $<10^{-15}$ | **`1D_CNN`** |
| **Bareilly** | Architecture (`consumption_only`) | `1D_CNN` vs. `CNN_LSTM` | $-13.3854$ | $<10^{-15}$ | $<10^{-15}$ | **`1D_CNN`** |
| **Bareilly** | Architecture (`consumption_only`) | `LSTM` vs. `CNN_LSTM` | $-8.2683$ | $3.44 \times 10^{-16}$ | $1.38 \times 10^{-15}$ | **`LSTM`** |
| **Bareilly** | Weather Ablation (`1D_CNN`) | `consumption_only` vs. `weather_retro` | $-14.1220$ | $<10^{-15}$ | $<10^{-15}$ | **`consumption_only`** |
| **Bareilly** | Weather Ablation (`1D_CNN`) | `consumption_only` vs. `weather_operational` | $-16.1782$ | $<10^{-15}$ | $<10^{-15}$ | **`consumption_only`** |

---

## 16. Interactive Research Dashboard & Prediction Simulator

The repository provides two interactive interfaces backed by the verified model checkpoints (`artifacts/checkpoints/agg_*.pt`) and empirical evaluation tables (`artifacts/metrics/*`):

1. **Midnight Grid 3D Glassmorphism Web Dashboard (`app/serve_dashboard.py` + `app/dashboard/`):**
   - **7 Interactive Views:** *Overview & Architecture*, *Aggregate Feeder Analysis*, *Individual Household Scale*, *Weather & Lag Regimes*, *Statistical Validation (DM Tests)*, *Methodology & Audit*, and the **Interactive Prediction Simulator**.
   - **Dual-Engine Prediction Simulator:** Executes live PyTorch inference via `/api/predict` when served through `python app/serve_dashboard.py`, and includes an exact Float32 client-side neural engine (`app/dashboard/simulator.js` + `simulator_bundle.js`, max discrepancy $8.34 \times 10^{-7}\text{ kW}$ vs. PyTorch) so all 18 aggregate models work even when `index.html` is opened directly as a static file.
2. **Streamlit Application (`app/app.py`):**
   - Provides interactive filtering across districts, spatial scales, feature configurations, and live 16-step demand simulation.

### Simulator Capabilities
- **Manual Entry:** Edit any of the 16 historical 15-minute load steps ($t-15 \dots t$) in physical $\text{kW}$.
- **6 Statistically Grounded Scenario Presets:** `Normal Consumption`, `High Demand`, `Low Demand`, `Morning Peak`, `Evening Peak`, and `Irregular Consumption`, sampled from the selected district's empirical training regime distribution.
- **CSV Upload:** Upload a custom CSV file ($\ge 16$ rows) with automatic column detection, duplicate-timestamp validation, and chronological sorting.
- **Multi-Model Comparison:** Run a single model or compare `1D_CNN`, `LSTM`, and `CNN_LSTM` side-by-side against Naive Persistence.

---

## 17. Repository Structure

```text
.
├── app/
│   ├── app.py                          # Streamlit interactive research & simulator app
│   ├── build_dashboard_data.py         # Compiles artifacts/metrics/* into app/dashboard/data.js
│   ├── build_simulator_bundle.py       # Exports Float32 weights & scalers to simulator_bundle.js
│   ├── serve_dashboard.py              # Hardened local HTTP server + /api/predict endpoints
│   └── dashboard/                      # Midnight Grid 3D Glassmorphism web interface
├── artifacts/
│   ├── checkpoints/                    # 18 verified aggregate PyTorch checkpoints (agg_*.pt)
│   └── metrics/                        # Saved JSON/CSV/Parquet results & comparison figure
├── AUDIT/                              # Technical, inference, security & release audit reports
├── config/
│   └── config.yaml                     # Master experimental configuration schema
├── data/
│   ├── README.md                       # Dataset provenance, DOIs & rebuild instructions
│   └── processed/                      # Tracked aggregate Parquets (mathura & bareilly)
├── docs/                               # Detailed methodology, datasets, models & limitations docs
├── notebooks/
│   ├── 01_eda_and_data_audit.ipynb
│   └── 02_colab_training_pipeline.ipynb
├── scripts/                            # Data processing, baseline, DM test & packaging scripts
├── src/                                # Core Python package (dataset, models, simulator, tests)
├── tests/                              # 44 unit & integration tests
├── .env.example                        # Optional server environment variable template
├── .gitignore
├── .python-version
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 18. Installation & Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd "CNN model"

# 2. Create and activate a virtual environment (Python 3.11+ recommended)
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 19. Running the Pipeline & Interactive Applications

### Launch the Web Dashboard & Interactive Prediction Simulator
```bash
python app/serve_dashboard.py
```
Open `http://127.0.0.1:8765` in your browser. (You can override host/port via `DASHBOARD_HOST` and `DASHBOARD_PORT` environment variables; see `.env.example`.)

### Launch the Streamlit Interface
```bash
streamlit run app/app.py
```

### Re-Run Statistical Tests or Rebuild Dashboard Bundles
```bash
python scripts/run_dm_tests.py
python app/build_dashboard_data.py
python app/build_simulator_bundle.py
```

---

## 20. Reproducibility & Verification Suite

Run the complete 44-test unit and integration suite covering sequence contiguity, $L = 16$ boundary purge gaps, training-only scaler isolation, metric formulas, Diebold–Mariano statistics, and live simulator inference across all 18 aggregate checkpoints:

```bash
python -m unittest discover tests -v
```

For full pipeline reproduction from raw smart-meter archives, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`data/README.md`](data/README.md).

---

## 21. Methodological & Operational Limitations

A detailed discussion of project limitations is provided in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md):
1. **Aggregate Feeder Scale Inversion:** All 18 aggregate deep learning models underperform Naive Persistence due to high 15-minute load inertia ($\rho_1 > 0.93$) and small sample size (`6,009–6,303` training windows).
2. **Weather Covariate Overfitting on Small Samples:** Hourly-interpolated ERA5-Land weather channels degrade aggregate accuracy relative to `consumption_only`, especially under the 120-hour operational lag.
3. **`RMSE` vs. `MAE` Trade-off on Individual Meters:** Huber loss (`delta = 1.0`) improves `RMSE`, $R^2$, and `Peak_MAE` across all 18 individual models, but yields slightly higher pooled `MAE` than Naive Persistence on Bareilly household meters (`0.1132–0.1154 kW` vs. `0.1114 kW`).
4. **Pending Phase 3.3 Individual Checkpoint Binary Export:** While all 18 individual household models were trained and evaluated on Colab GPU (metrics logged in `artifacts/metrics/training_results.json`), the binary `ind_*.pt` files and pointwise `individual_predictions.parquet` remain pending export from Colab; therefore, live simulator inference is restricted to the 18 verified aggregate feeder checkpoints (`agg_*.pt`).

---

## 22. Documentation Index

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — Mathematical formulation, contiguity rules, split/purge protocol, loss function, and Diebold–Mariano testing
- [`docs/DATASETS.md`](docs/DATASETS.md) — CEEW smart-meter cohorts, IIIT Delhi exploratory check, and ERA5-Land weather interpolation
- [`docs/MODEL_ARCHITECTURES.md`](docs/MODEL_ARCHITECTURES.md) — Layer-by-layer specifications and parameter counts for `1D_CNN`, `LSTM`, and `CNN_LSTM`
- [`docs/INFERENCE_GUIDE.md`](docs/INFERENCE_GUIDE.md) — Interactive Prediction Simulator guide and Python API reference
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — Deterministic seeding and step-by-step reproduction instructions
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — Detailed analysis of scale inversion, weather lag effects, and artifact boundaries
- [`docs/REFERENCES.md`](docs/REFERENCES.md) — Bibliographic references and dataset DOIs

---

## 23. References, License & Citation

- **Code License:** Released under the [MIT License](LICENSE). External datasets (CEEW Smart Meter Data, IIIT Delhi Residential Power Data, and Copernicus ERA5-Land) remain governed by their upstream licenses documented in [`LICENSE`](LICENSE) and [`data/README.md`](data/README.md).
- **Citation:** See [`CITATION.cff`](CITATION.cff) and [`docs/REFERENCES.md`](docs/REFERENCES.md).
