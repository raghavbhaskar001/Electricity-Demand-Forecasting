# Experimental Methodology

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala

---

## 1. Forecasting Problem Formulation

Let $y_t \in \mathbb{R}_{\ge 0}$ denote the average electrical power demand (in physical kilowatts, $\text{kW}$) over a 15-minute interval indexed by $t$, and let $\mathbf{x}_t \in \mathbb{R}^{C}$ denote the feature vector at step $t$, where $C \in \{1, 10\}$ depends on the selected covariate configuration (`consumption_only`, `consumption_weather_retro`, or `consumption_weather_operational`).

Given a contiguous historical lookback window of $L = 16$ fifteen-minute intervals ($4.0\text{ hours}$):

$$
\mathbf{X}_{t-L+1:t} = \begin{bmatrix} \mathbf{x}_{t-15}^{\top} \\ \mathbf{x}_{t-14}^{\top} \\ \vdots \\ \mathbf{x}_{t}^{\top} \end{bmatrix} \in \mathbb{R}^{16 \times C}
$$

each model $f_{\theta} : \mathbb{R}^{16 \times C} \to \mathbb{R}$ predicts the one-step-ahead ($H = 1$, $+15\text{ min}$) electricity demand $\hat{y}_{t+1}$.

---

## 2. Temporal Resampling & Contiguous Window Construction

1. **Sub-Interval Completeness Rule ($5 \times 3\text{-min} \to 15\text{-min}$):**  
   Raw CEEW smart-meter measurements are logged at 3-minute resolution (`t_kWh`). Rather than imputing missing readings across telemetry dropouts, a 15-minute interval is marked valid only if all 5 constituent 3-minute observations are present (`valid_obs_count == 5`). Mean active power in kilowatts is computed as:
   $$
   P_{15\text{min}}\,(\text{kW}) = \frac{1}{5} \sum_{j=1}^{5} \frac{E_{3\text{min}, j}\,(\text{kWh})}{3 / 60\text{ h}} = 4 \sum_{j=1}^{5} E_{3\text{min}, j}\,(\text{kWh})
   $$
2. **Fixed Cohort Selection ($\ge 90\%$ Coverage):**  
   To prevent artificial step changes in aggregate feeder demand caused by meters entering or leaving the sample mid-year, aggregate series are constructed strictly from meters maintaining $\ge 90\%$ temporal completeness across the full observation span (`38` meters in Mathura, `46` meters in Bareilly). Furthermore, an aggregate 15-minute timestamp is retained for primary modelling (`is_clean_agg_primary == True`) only when at least $90\%$ of the fixed cohort reports valid 5-of-5 readings during that interval.
3. **Strict Contiguity Enforcement:**  
   In `src/dataset.py` (`extract_contiguous_sequences`), a sliding window $(t-15, \dots, t \to t+1)$ is emitted if and only if all $L + H = 17$ consecutive timestamps are separated by exactly $\Delta t = 15\text{ minutes}$ with no NaN values in any input channel or target. Windows spanning telemetry outages are discarded rather than stitched across gaps.

---

## 3. Chronological Partitioning & Boundary Purge Protocol

Every contiguous sequence index is partitioned strictly in chronological order ($70\%$ Training, $15\%$ Validation, $15\%$ Test) with an explicit purge gap of $L = 16$ sequences removed at both split boundaries (`Train -> Val` and `Val -> Test`):

- **Aggregate Series (`split_contiguous_sequences`):**
  - **Mathura Aggregate (`9,005` total sequences):** `Train = 6,303`, `Purge 1 = 16`, `Val = 1,350`, `Purge 2 = 16`, `Test = 1,320`.
  - **Bareilly Aggregate (`8,585` total sequences):** `Train = 6,009`, `Purge 1 = 16`, `Val = 1,287`, `Purge 2 = 16`, `Test = 1,257`.
- **Individual Household Panels (`create_individual_meter_datasets`):**
  - Chronological $70/15/15$ partitioning with $L = 16$ boundary purge is applied **independently per household (`meter_id`)** before pooling across meters so every household is represented in train, validation, and test partitions without cross-boundary window overlap:
  - **Mathura Individual (`38` meters, `1,536,402` sequences):** `Train = 1,075,464`, `Purge = 1,216` (`38 x 32`), `Val = 230,441`, `Test = 229,281`.
  - **Bareilly Individual (`46` meters, `2,668,842` sequences):** `Train = 1,868,167`, `Purge = 1,472` (`46 x 32`), `Val = 400,306`, `Test = 398,897`.

---

## 4. Feature Scaling & Inverse Transformation

- **Aggregate Scale:** A single `StandardScaler` (`scaler_X`) for features and `StandardScaler` (`scaler_y`) for the target are fit **exclusively on unique timestamps belonging to the training partition**.
- **Individual Household Scale:** Separate `StandardScaler` instances (`scalers_X[meter_id]`, `scalers_y[meter_id]`) are fit **exclusively on each household's training timestamps** so high-consumption households do not distort the z-score normalization of low-consumption households.
- **Physical-Unit Evaluation (`kW`):** Before computing any accuracy metric, model outputs are inverse-transformed back to physical kilowatts:
  $$
  \hat{y}_{t+1}^{(\text{kW})} = \hat{z}_{t+1} \cdot \sigma_{y,\text{train}} + \mu_{y,\text{train}}
  $$

---

## 5. Loss Function & Optimization Schedule

All 36 experiments are trained with identical optimization settings (`src/train_runner.py`):
- **Objective Function:** Huber Loss (`nn.HuberLoss(delta=1.0)`) on standardized target values, combining quadratic sensitivity for small errors ($|e| \le 1\sigma$) with linear robustness against extreme switching transients ($|e| > 1\sigma$).
- **Optimizer:** `AdamW` ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$).
- **Gradient Clipping:** Global $\ell_2$-norm clipped at `max_norm = 1.0` per batch.
- **Learning Rate Scheduler:** `ReduceLROnPlateau(mode='min', factor=0.5, patience=4, min_lr=1e-6)` monitoring validation Huber loss.
- **Batch Size & Epoch Budget:**
  - Aggregate feeder runs: `batch_size = 64`, `max_epochs = 40`, `early_stopping_patience = 8`.
  - Individual household runs: `batch_size = 512`, `max_epochs = 15`, `early_stopping_patience = 6`.

---

## 6. Evaluation Metrics & Statistical Hypothesis Testing

All metrics (`src/evaluation.py`) are computed in physical $\text{kW}$:
- **RMSE ($\text{kW}$):** $\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$
- **MAE ($\text{kW}$):** $\frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$
- **WAPE ($\%$):** $100 \times \frac{\sum_{i=1}^{N}|y_i - \hat{y}_i|}{\sum_{i=1}^{N}|y_i|}$ (primary percentage error metric; robust to zero-load intervals)
- **$R^2$:** $1 - \frac{\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}(y_i - \bar{y})^2}$
- **Persistence Skill Score ($\%$):** $100 \times \left(1 - \frac{\text{MSE}_{\text{model}}}{\text{MSE}_{\text{persistence}}}\right)$, where Naive Persistence predicts $\hat{y}_{t+1}^{\text{pers}} = y_t$.
- **Peak Load MAE ($\text{kW}$):** MAE restricted to the top $5\%$ highest-demand test intervals ($y_i \ge P_{95}$).

### Diebold–Mariano (DM) Test with Harvey–Leybourne–Newbold Correction
For aggregate feeder models (`src/statistical_tests.py`, `scripts/run_dm_tests.py`), pairwise forecast comparisons evaluate squared-error loss differentials $d_t = e_{1,t}^2 - e_{2,t}^2$ using Newey–West Bartlett-kernel HAC variance estimation and the Harvey–Leybourne–Newbold (1997) finite-sample Student-$t$ adjustment (`df = T - 1`). Family-wise Type I error across each 12-comparison district family is controlled at $\alpha = 0.05$ using the Holm–Bonferroni step-down procedure.
