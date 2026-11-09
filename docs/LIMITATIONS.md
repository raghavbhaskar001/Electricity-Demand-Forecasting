# Methodological & Operational Limitations

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala

This document records the empirical, statistical, and artifact-level limitations established during the project audit.

---

## 1. Scale Inversion: Why Deep Learning Underperforms Naive Persistence on Aggregate Feeders

On the smooth 15-minute aggregate feeder series (`Mathura N = 38` meters, `Bareilly N = 46` meters), **all 18 deep learning models underperform the zero-parameter Naive Persistence baseline** ($\hat{y}_{t+1} = y_t$):
- **Mathura Aggregate:** Naive Persistence achieves `RMSE = 1.2033 kW` (`MAE = 0.8713 kW`, `WAPE = 11.22%`, $R^2 = 0.8825$), whereas the best neural model (`LSTM` with `consumption_only`) achieves `RMSE = 1.5298 kW` (`Skill = -61.64%`, Diebold–Mariano $p_{\text{adj}} < 10^{-15}$ in favor of persistence).
- **Bareilly Aggregate:** Naive Persistence achieves `RMSE = 2.3063 kW` (`MAE = 1.7289 kW`, `WAPE = 8.97%`, $R^2 = 0.8452$), whereas the best neural model (`1D_CNN` with `consumption_only`) achieves `RMSE = 2.9540 kW` (`Skill = -64.05%`, Diebold–Mariano $p_{\text{adj}} = 7.47 \times 10^{-10}$ in favor of persistence).

**Root Cause:** Summing 38–46 residential meters smooths out individual appliance switching transients by the Law of Large Numbers, producing a high-inertia series with lag-1 autocorrelation $\rho_1 > 0.93$. With only `6,009–6,303` training sequences, neural networks with `52K–71K` parameters have roughly $8.5\text{–}11.5$ parameters per training example and cannot beat the identity mapping $\hat{y}_{t+1} = y_t$ out-of-sample.

---

## 2. Weather Covariate Degradation on Small-Sample Aggregate Series

Adding 9 ERA5-Land meteorological channels (`consumption_weather_retro` or `consumption_weather_operational`) consistently degrades aggregate feeder accuracy relative to `consumption_only`:
- In Mathura, `LSTM` test RMSE worsens from `1.5298 kW` (`consumption_only`) to `1.8819 kW` (`retro`) and `2.5015 kW` (`operational`).
- In Bareilly, `1D_CNN` test RMSE worsens from `2.9540 kW` (`consumption_only`) to `4.2940 kW` (`retro`) and `4.4628 kW` (`operational`).

**Root Cause:** Three compounding mechanisms explain this degradation:
1. **Temporal Resolution Mismatch:** ERA5-Land is native hourly (`60-min`) reanalysis interpolated to `15-min` steps; over a single 15-minute step, ambient weather changes negligibly and adds collinearity rather than high-frequency signal.
2. **Spatial Centroid Offset:** ERA5-Land grid points represent ~9 km regional averages rather than feeder-level microclimates.
3. **120-Hour Operational Publication Lag:** Under `consumption_weather_operational`, the 5-day release latency of verified ERA5-Land data decouples synoptic weather states from current load, inducing severe validation/test overfitting on small aggregate sample sizes.

---

## 3. Metric Divergence (`RMSE` vs `MAE`) on Individual Household Meters

At the disaggregated household scale (`1.08M` training windows in Mathura, `1.87M` in Bareilly), **all 18 deep learning models outperform Naive Persistence on `RMSE`, $R^2$, and `Peak_MAE`**:
- **Mathura Individual (`229,281` test sequences):** `CNN_LSTM` (`consumption_weather_operational`) reduces pooled RMSE from `0.2110 kW` to `0.1880 kW` (`+20.57%` skill) and improves $R^2$ from `0.6002` to `0.6826`. Seven of nine Mathura models also improve on `MAE` (`0.0671 kW` vs `0.0707 kW`).
- **Bareilly Individual (`398,897` test sequences):** `LSTM` (`consumption_weather_operational`) reduces pooled RMSE from `0.2755 kW` to `0.2548 kW` (`+14.40%` skill) and improves $R^2$ from `0.7425` to `0.7796`. However, **0 of 9 Bareilly models improve on pooled `MAE`** (`0.1132–0.1154 kW` vs persistence `0.1114 kW`).

**Root Cause:** Training under `HuberLoss(delta=1.0)` penalizes large appliance-switching spikes ($|e| > 1\sigma$), pulling step-change predictions toward the conditional expectation. This strongly reduces squared errors (`RMSE`) during abrupt turn-on/turn-off events, but introduces a `2–4 W` conditional-mean bias during long flat standby periods where Naive Persistence ($\hat{y}_{t+1} = y_t$) incurs nearly zero absolute error.

---

## 4. Pending Phase 3.3 Export of Individual Household Checkpoints (`.pt`)

While all 18 individual household models completed training and test evaluation on Google Colab GPU (with full epoch histories, pooled test metrics, and per-meter quartiles recorded in `artifacts/metrics/training_results.json`), the **18 binary `.pt` weight files (`ind_*.pt`) and pointwise individual test predictions (`individual_predictions.parquet`) have not yet been exported from the Colab runtime to local storage**:
- Consequently, formal panel Diebold–Mariano and Wilcoxon signed-rank p-values are not claimed for the individual household tier.
- Live interactive simulator execution is restricted to the **18 locally verified aggregate feeder checkpoints (`agg_*.pt`)**.

---

## 5. Lookback Horizon Scope ($L = 16$ Steps = 4.0 Hours)

The sequence window length $L = 16$ (`4.0 hours`) captures short-term autoregressive momentum, intra-day load ramps, and appliance cycle durations, but does not directly span a 24-hour (`96-step`) diurnal cycle or 7-day (`672-step`) weekly cycle inside the raw sequence tensor.

---

## 6. Exclusion of IIIT Delhi Cohort from Deep Learning Training

The processed IIIT Delhi series (`data/processed/delhi_features.parquet`) contains only `370` fifteen-minute intervals (`96` contiguous 16-step windows). After chronological partitioning and $L = 16$ boundary purging, `0` test windows remain; therefore, New Delhi is documented strictly as an exploratory descriptive dataset.
