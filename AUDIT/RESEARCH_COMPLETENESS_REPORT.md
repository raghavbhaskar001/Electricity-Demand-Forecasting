# Research Completeness and Result Verification Report

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala  
**Audit Date:** 2026-11-09  

---

## 1. Experimental Matrix Verification (36 / 36 Experiments)

The study evaluates a locked $2 \times 2 \times 3 \times 3 = 36$ factorial design across:
- **Forecasting Scales (2):** Aggregate feeder cohort (`aggregate`) and Individual household pooled (`individual_pooled`).
- **Districts (2):** Mathura (`38` active household meters) and Bareilly (`46` active household meters) from the CEEW Smart Meter dataset (`doi:10.7910/DVN/GOCHJH`).
- **Model Architectures (3):** `1D_CNN` (`CNN1DForecaster`), `LSTM` (`LSTMForecaster`), and `CNN_LSTM` (`CNNLSTMForecaster`).
- **Feature Configurations (3):**
  - `consumption_only` (`1` feature: `demand_target`)
  - `consumption_weather_retro` (`10` features: `demand_target` + 9 contemporaneous ERA5-Land variables, $t_{\text{weather}} \le T$)
  - `consumption_weather_operational` (`10` features: `demand_target` + 9 operational ERA5-Land variables with 120-hour availability lag, $t_{\text{weather}} \le T - 120\text{ h}$)

### Artifact Availability Summary Across All 36 Experiments

| Dimension | Aggregate Scale (18 Experiments) | Individual Pooled Scale (18 Experiments) |
| :--- | :--- | :--- |
| **Training Completion** | **18 / 18 Complete** (local CPU/GPU, max 40 epochs, early stopping patience 8) | **18 / 18 Complete** (Google Colab T4 GPU, max 15 epochs, early stopping patience 6) |
| **Logged Test Metrics (`training_results.json`)** | **18 / 18 Available** | **18 / 18 Available** (pooled metrics + per-meter test metric distributions across all 38 Mathura and 46 Bareilly meters) |
| **Epoch Training Logs (`artifacts/metrics/training_logs/`)** | **18 / 18 Available** | **0 / 18 Local** (remained in Colab runtime) |
| **Finalized `.pt` Checkpoints (`artifacts/checkpoints/`)** | **18 / 18 Verified** (`agg_*.pt`) | **0 / 18 Verified Locally** (`ind_mathura_1D_CNN_consumption_only.pt` is an interrupted epoch-3 local partial run; 17 others pending Colab export) |
| **Pointwise Test Predictions (`.parquet`)** | **18 / 18 Verified** (`aggregate_test_predictions.parquet`, `23,733` total test rows: `1,320` Mathura × 9 + `1,257` Bareilly × 9) | **0 / 18 Available Locally** (pending Colab export of finalized `.pt` checkpoints) |
| **Preprocessing / Scaler Reproducibility** | **18 / 18 Verified** (`data/processed/{district}_features.parquet` + deterministic 70% train split `StandardScaler`) | **18 / 18 Reproducible from `*_individual_meters.parquet`**, but blocked from live inference without finalized `.pt` weights |
| **Statistical Hypothesis Testing** | **Validated** (Diebold–Mariano with Newey–West HAC $K=8$, HLN small-sample adjustment, Holm step-down correction across 7 hypotheses per district + HAC sensitivity $K \in \{1,2,4,8,12,16\}$) | **Descriptive Only — Panel Tests Pending** (Wilcoxon signed-rank and meter-blocked bootstrap scripts implemented in `src/panel_statistical_tests.py`, awaiting pointwise prediction export) |

---

## 2. Complete 36-Experiment Verification Table

All values below are verified directly from `artifacts/metrics/training_results.json`, `artifacts/metrics/baseline_benchmarks.json`, and `artifacts/metrics/phase_3_4_consolidated_results.csv`. All error metrics are in physical **kW** (`P = 4 × E_15min`). Classification metrics (`Accuracy`, `Precision`, `Recall`, `F1`, `Confusion Matrix`) are not applicable to continuous load regression and are not used.

### 2.1 Aggregate Feeder Cohort (18 Experiments + Baselines)

- **Mathura Aggregate Census:** `9,005` contiguous windows (`Train = 6,303`, `Purge 1 = 16`, `Val = 1,350`, `Purge 2 = 16`, `Test = 1,320`; test span `2019-10-13 18:30` to `2020-02-19 23:45`).
- **Bareilly Aggregate Census:** `8,585` contiguous windows (`Train = 6,009`, `Purge 1 = 16`, `Val = 1,287`, `Purge 2 = 16`, `Test = 1,257`; test span `2020-05-12 18:45` to `2020-08-23 23:45`).

| Experiment ID | District | Architecture | Feature Config | Best / Total Epochs | RMSE (kW) | MAE (kW) | WAPE (%) | $R^2$ | Persistence Skill (%) | Checkpoint | Pointwise Preds | Validation Status |
| :--- | :--- | :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | :--- |
| *Naive Persistence Baseline* | Mathura | $y_{t}$ | — | — | **1.2033** | **0.8713** | **11.22** | **0.8825** | **0.00%** | — | Verified | Baseline |
| `agg_mathura_LSTM_consumption_only` | Mathura | LSTM | `consumption_only` | 15 / 23 | **1.5298** | **1.2797** | **16.48** | **0.8101** | **-61.64%** | Verified | Verified | DM-Validated |
| `agg_mathura_1D_CNN_consumption_only` | Mathura | 1D CNN | `consumption_only` | 8 / 16 | 1.5490 | 1.2885 | 16.60 | 0.8054 | -65.71% | Verified | Verified | DM-Validated |
| `agg_mathura_LSTM_consumption_weather_operational` | Mathura | LSTM | `weather_operational` | 11 / 19 | 2.1292 | 1.8574 | 23.92 | 0.6322 | -213.12% | Verified | Verified | DM-Validated |
| `agg_mathura_CNN_LSTM_consumption_only` | Mathura | CNN–LSTM | `consumption_only` | 20 / 28 | 2.2417 | 1.9086 | 24.58 | 0.5923 | -247.06% | Verified | Verified | DM-Validated |
| `agg_mathura_LSTM_consumption_weather_retro` | Mathura | LSTM | `weather_retro` | 7 / 15 | 2.5153 | 2.1960 | 28.29 | 0.4868 | -336.96% | Verified | Verified | DM-Validated |
| `agg_mathura_1D_CNN_consumption_weather_operational` | Mathura | 1D CNN | `weather_operational` | 12 / 20 | 2.7396 | 2.3931 | 30.83 | 0.3911 | -418.36% | Verified | Verified | DM-Validated |
| `agg_mathura_CNN_LSTM_consumption_weather_retro` | Mathura | CNN–LSTM | `weather_retro` | 11 / 19 | 2.8226 | 2.4645 | 31.75 | 0.3537 | -450.23% | Verified | Verified | DM-Validated |
| `agg_mathura_CNN_LSTM_consumption_weather_operational` | Mathura | CNN–LSTM | `weather_operational` | 11 / 19 | 3.2179 | 2.8312 | 36.47 | 0.1600 | -615.17% | Verified | Verified | DM-Validated |
| `agg_mathura_1D_CNN_consumption_weather_retro` | Mathura | 1D CNN | `weather_retro` | 22 / 30 | 3.6498 | 3.2379 | 41.71 | -0.0806 | -820.02% | Verified | Verified | DM-Validated |
| *Naive Persistence Baseline* | Bareilly | $y_{t}$ | — | — | **2.3063** | **1.7289** | **8.97** | **0.8452** | **0.00%** | — | Verified | Baseline |
| `agg_bareilly_1D_CNN_consumption_only` | Bareilly | 1D CNN | `consumption_only` | 14 / 22 | **2.9540** | **2.2348** | **11.60** | **0.7461** | **-64.05%** | Verified | Verified | DM-Validated |
| `agg_bareilly_LSTM_consumption_only` | Bareilly | LSTM | `consumption_only` | 12 / 20 | 3.8015 | 2.7655 | 14.35 | 0.5795 | -171.69% | Verified | Verified | DM-Validated |
| `agg_bareilly_LSTM_consumption_weather_operational` | Bareilly | LSTM | `weather_operational` | 5 / 13 | 4.0785 | 2.9855 | 15.49 | 0.5159 | -212.73% | Verified | Verified | DM-Validated |
| `agg_bareilly_LSTM_consumption_weather_retro` | Bareilly | LSTM | `weather_retro` | 5 / 13 | 4.1482 | 3.0338 | 15.74 | 0.4993 | -223.51% | Verified | Verified | DM-Validated |
| `agg_bareilly_CNN_LSTM_consumption_only` | Bareilly | CNN–LSTM | `consumption_only` | 24 / 32 | 4.8971 | 3.4768 | 18.04 | 0.3021 | -350.86% | Verified | Verified | DM-Validated |
| `agg_bareilly_1D_CNN_consumption_weather_retro` | Bareilly | 1D CNN | `weather_retro` | 20 / 28 | 5.1299 | 4.0444 | 20.99 | 0.2342 | -394.74% | Verified | Verified | DM-Validated |
| `agg_bareilly_CNN_LSTM_consumption_weather_retro` | Bareilly | CNN–LSTM | `weather_retro` | 14 / 22 | 5.2069 | 3.7749 | 19.59 | 0.2111 | -409.70% | Verified | Verified | DM-Validated |
| `agg_bareilly_CNN_LSTM_consumption_weather_operational` | Bareilly | CNN–LSTM | `weather_operational` | 14 / 22 | 5.3707 | 3.9837 | 20.67 | 0.1606 | -442.28% | Verified | Verified | DM-Validated |
| `agg_bareilly_1D_CNN_consumption_weather_operational` | Bareilly | 1D CNN | `weather_operational` | 8 / 16 | 5.5341 | 4.4781 | 23.24 | 0.1088 | -475.78% | Verified | Verified | DM-Validated |

### 2.2 Individual Household Pooled Scale (18 Experiments + Baselines)

- **Mathura Individual Census (`38` meters):** `1,536,402` contiguous windows (`Train = 1,075,464`, `Purge = 1,216`, `Val = 230,441`, `Test = 229,281`).
- **Bareilly Individual Census (`46` meters):** `2,668,842` contiguous windows (`Train = 1,868,167`, `Purge = 1,472`, `Val = 400,306`, `Test = 398,897`).

| Experiment ID | District | Architecture | Feature Config | Best / Total Epochs | Pooled RMSE (kW) | Pooled MAE (kW) | Pooled WAPE (%) | Pooled $R^2$ | Pooled Skill (%) | Median Meter RMSE (IQR) | Median Meter Skill (%) | Checkpoint | Validation Status |
| :--- | :--- | :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: | :--- |
| *Naive Persistence Baseline* | Mathura | $y_{t}$ | — | — | 0.2110 | 0.0707 | 38.13 | 0.6002 | 0.00% | — | — | — | Baseline |
| `ind_mathura_CNN_LSTM_consumption_weather_operational` | Mathura | CNN–LSTM | `weather_operational` | 11 / 15 | **0.1880** | 0.0680 | 36.74 | **0.6826** | **+20.57%** | 0.1311 (0.1456) | +16.56% | Missing (Colab) | Descriptive · Pending |
| `ind_mathura_CNN_LSTM_consumption_weather_retro` | Mathura | CNN–LSTM | `weather_retro` | 8 / 14 | 0.1886 | **0.0671** | **36.19** | 0.6804 | +20.06% | **0.1305** (0.1419) | +16.13% | Missing (Colab) | Descriptive · Pending |
| `ind_mathura_LSTM_consumption_weather_retro` | Mathura | LSTM | `weather_retro` | 10 / 15 | 0.1898 | 0.0721† | 38.89† | 0.6765 | +19.07% | 0.1311 (0.1463) | +15.40% | Missing (Colab) | Descriptive · Pending |
| `ind_mathura_LSTM_consumption_weather_operational` | Mathura | LSTM | `weather_operational` | 12 / 15 | 0.1899 | 0.0675 | 36.48 | 0.6761 | +18.95% | 0.1311 (0.1436) | +16.53% | Missing (Colab) | Descriptive · Pending |
| `ind_mathura_1D_CNN_consumption_weather_operational` | Mathura | 1D CNN | `weather_operational` | 8 / 14 | 0.1899 | 0.0714† | 38.59† | 0.6760 | +18.91% | 0.1329 (0.1471) | +15.89% | Missing (Colab) | Descriptive · Pending |
| `ind_mathura_CNN_LSTM_consumption_only` | Mathura | CNN–LSTM | `consumption_only` | 13 / 15 | 0.1901 | 0.0696 | 37.54 | 0.6754 | +18.80% | 0.1340 (0.1469) | +16.08% | Missing (Colab) | Descriptive · Pending |
| `ind_mathura_LSTM_consumption_only` | Mathura | LSTM | `consumption_only` | 11 / 15 | 0.1906 | 0.0690 | 37.22 | 0.6737 | +18.37% | 0.1358 (0.1476) | +13.30% | Missing (Colab) | Descriptive · Pending |
| `ind_mathura_1D_CNN_consumption_weather_retro` | Mathura | 1D CNN | `weather_retro` | 12 / 15 | 0.1914 | 0.0680 | 36.69 | 0.6710 | +17.70% | 0.1328 (0.1432) | +15.68% | Missing (Colab) | Descriptive · Pending |
| `ind_mathura_1D_CNN_consumption_only` | Mathura | 1D CNN | `consumption_only` | 11 / 15 | 0.1920 | 0.0703 | 37.92 | 0.6687 | +17.14% | 0.1387 (0.1489) | +13.87% | Partial (ep 3) | Descriptive · Pending |
| *Naive Persistence Baseline* | Bareilly | $y_{t}$ | — | — | 0.2755 | **0.1114** | **27.22** | 0.7425 | 0.00% | — | — | — | Baseline |
| `ind_bareilly_LSTM_consumption_weather_operational` | Bareilly | LSTM | `weather_operational` | 14 / 15 | **0.2548** | 0.1137† | 27.79† | **0.7796** | **+14.40%** | 0.1384 (0.1341) | **+14.25%** | Missing (Colab) | Descriptive · Pending |
| `ind_bareilly_CNN_LSTM_consumption_weather_operational` | Bareilly | CNN–LSTM | `weather_operational` | 13 / 15 | 0.2550 | 0.1154† | 28.20† | 0.7794 | +14.30% | **0.1358** (0.1332) | +13.45% | Missing (Colab) | Descriptive · Pending |
| `ind_bareilly_CNN_LSTM_consumption_weather_retro` | Bareilly | CNN–LSTM | `weather_retro` | 15 / 15 | 0.2551 | 0.1140† | 27.85† | 0.7792 | +14.26% | **0.1358** (0.1332) | +14.01% | Missing (Colab) | Descriptive · Pending |
| `ind_bareilly_LSTM_consumption_weather_retro` | Bareilly | LSTM | `weather_retro` | 14 / 15 | 0.2553 | 0.1132† | 27.65† | 0.7788 | +14.08% | 0.1373 (0.1328) | +13.73% | Missing (Colab) | Descriptive · Pending |
| `ind_bareilly_1D_CNN_consumption_weather_retro` | Bareilly | 1D CNN | `weather_retro` | 7 / 13 | 0.2565 | 0.1141† | 27.88† | 0.7768 | +13.29% | 0.1379 (0.1335) | +13.22% | Missing (Colab) | Descriptive · Pending |
| `ind_bareilly_1D_CNN_consumption_weather_operational` | Bareilly | 1D CNN | `weather_operational` | 7 / 13 | 0.2570 | 0.1141† | 27.89† | 0.7759 | +12.96% | 0.1380 (0.1353) | +12.91% | Missing (Colab) | Descriptive · Pending |
| `ind_bareilly_LSTM_consumption_only` | Bareilly | LSTM | `consumption_only` | 14 / 15 | 0.2576 | 0.1147† | 28.04† | 0.7748 | +12.54% | 0.1375 (0.1332) | +12.84% | Missing (Colab) | Descriptive · Pending |
| `ind_bareilly_CNN_LSTM_consumption_only` | Bareilly | CNN–LSTM | `consumption_only` | 13 / 15 | 0.2576 | 0.1133† | 27.68† | 0.7748 | +12.51% | 0.1386 (0.1339) | +12.15% | Missing (Colab) | Descriptive · Pending |
| `ind_bareilly_1D_CNN_consumption_only` | Bareilly | 1D CNN | `consumption_only` | 5 / 11 | 0.2591 | 0.1150† | 28.11† | 0.7722 | +11.54% | 0.1391 (0.1357) | +11.84% | Missing (Colab) | Descriptive · Pending |

*Note (†): Indicates models where pooled MAE/WAPE is slightly higher than Naive Persistence even though RMSE and $R^2$ improve over Naive Persistence.*

---

## 3. Critical Scientific Distinctions Verified

### 3.1 Scale Inversion Between Aggregate Feeder and Individual Household Forecasting
- **Aggregate Feeder Scale (Negative Persistence Skill across all 18 models):**
  - Summing 38–46 residential meters smooths high-frequency appliance switching into a high-autocorrelation trajectory where 15-minute Naive Persistence ($\hat{y}_{t+1} = y_t$) already achieves $R^2 = 0.8825$ (`RMSE = 1.2033 kW`) in Mathura and $R^2 = 0.8452$ (`RMSE = 2.3063 kW`) in Bareilly.
  - With only ~6,000–6,300 training windows, all 18 aggregate deep learning models underperform Naive Persistence (`Skill = -61.64%` to `-820.02%`), confirmed as statistically significant via Diebold–Mariano tests (`DM = +9.972, p_adj < 1e-15` in Mathura; `DM = +6.315, p_adj = 7.47e-10` in Bareilly).
  - Adding 9 meteorological covariates to the 1-feature aggregate series increases input dimensionality from `16×1` to `16×10` under small sample size (`N_train ≈ 6,000`), inducing severe covariate overfitting and seasonal drift between the training split (summer/monsoon) and test split (winter in Mathura).
- **Individual Household Scale (Positive RMSE Persistence Skill across all 18 models, Mixed MAE):**
  - Individual household demand exhibits low 15-minute autocorrelation and abrupt appliance step changes (`Persistence R² = 0.6002` in Mathura, `0.7425` in Bareilly), while pooling across meters provides $1.08\text{M}$ (Mathura) and $1.87\text{M}$ (Bareilly) training windows.
  - All 18 individual models achieve positive RMSE Persistence Skill (`+17.14%` to `+20.57%` in Mathura; `+11.54%` to `+14.40%` in Bareilly) and higher $R^2$.
  - **MAE vs. RMSE Divergence:** While RMSE and $R^2$ improve for all 18 individual models, **MAE** improves for 7/9 Mathura models (`0.0671–0.0703 kW` vs `0.0707 kW` baseline) and **0/9 Bareilly models** (`0.1132–0.1154 kW` vs `0.1114 kW` baseline). Because neural networks trained with `HuberLoss(delta=1.0)` pull predictions toward the conditional mean during volatile transitions, they substantially penalize large squared errors (reducing RMSE) while introducing a small `2–4 W` residual offset during flat standby periods where $y_{t+1} = y_t$ is exact.

---

## 4. Documented Limitations & Configuration Discrepancies

1. **Missing Finalized Individual Checkpoints & Pointwise Predictions (Phase 3.3 Pending):**
   - Only 1 individual `.pt` file exists locally (`ind_mathura_1D_CNN_consumption_only.pt`), and inspection confirms it is an interrupted epoch-3 partial checkpoint (`val_loss = 0.124437` vs finalized Colab epoch-11 `0.121401`). The 18 finalized individual checkpoints and pointwise predictions remain pending export from Google Colab.
2. **Individual Statistical Tests Pending:**
   - Because pointwise individual test predictions are not exported locally, meter-level Wilcoxon signed-rank and block-bootstrap tests cannot be executed yet; all individual-scale comparisons are reported strictly as descriptive benchmark results.
3. **IIIT Delhi Exploratory Cohort (`96` Contiguous Sequences):**
   - After enforcing strict 16-step contiguity (`pctmissing <= 10%`, `19/19` apartments reporting), `data/processed/delhi_features.parquet` yields only 96 contiguous windows (`67` train, `13` val, `0` test after two 16-step boundary purges). It is documented as exploratory/descriptive only and excluded from deep learning training.
4. **Minor Schema Drift in `config/config.yaml` (Identified for Fix in Phase 7/8):**
   - `config/config.yaml` listed `consumption_weather_retrospective` and generic placeholder training defaults (`epochs: 50`, `patience: 5`, `early_stopping_patience: 10`), whereas `src/dataset.py` and `src/train_runner.py` executed `consumption_weather_retro`, `epochs_agg=40`, `epochs_ind=15`, `batch_size_agg=64`, `batch_size_ind=512`, `ReduceLROnPlateau(patience=4)`, and `early_stopping_patience=8` (aggregate) / `6` (individual). `config/config.yaml` will be updated to match the exact executed code in `src/train_runner.py`.
