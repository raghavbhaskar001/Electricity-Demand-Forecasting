# Interactive Prediction Simulator & Live Inference Guide

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala

---

## 1. Overview

The repository includes an **Interactive Prediction Simulator** (`src/simulator.py`, `app/serve_dashboard.py`, `app/dashboard/simulator.js`, and `app/app.py`) that executes real-time next-15-minute ($H = 1$) demand forecasting across all **18 verified aggregate feeder checkpoints** (`2 districts × 3 architectures × 3 feature sets`).

Every forecast executes the complete inference pipeline:
1. Validates the 16-step lookback sequence ($4.0\text{ hours}$ at 15-minute intervals, in physical $\text{kW}$).
2. Assembles the exact feature matrix $\mathbf{X} \in \mathbb{R}^{16 \times C}$ ($C = 1$ for `consumption_only`; $C = 10$ when paired with historical district weather contexts).
3. Applies the training-partition `StandardScaler` (`scaler_X`).
4. Runs the forward pass in evaluation mode (`model.eval()`, `torch.no_grad()`).
5. Inverse-transforms the output via `scaler_y` back into physical $\text{kW}$ and compares against the Naive Persistence baseline ($\hat{y}_{t+1}^{\text{pers}} = y_t$).

> **Scope Note:** All 18 aggregate feeder checkpoints (`agg_*.pt`) are verified and bundled locally in `artifacts/checkpoints/`. Individual household checkpoints (`ind_*.pt`) are marked unavailable in the simulator until Phase 3.3 Colab checkpoint export is completed.

---

## 2. Launching the Interactive Simulator

### Option A — Glassmorphism Web Dashboard + Live Python API (`serve_dashboard.py`)
```bash
python app/serve_dashboard.py
```
Then open `http://127.0.0.1:8765` in your browser and select the **Prediction Simulator** tab in the top navigation bar.

### Option B — Streamlit Application (`app/app.py`)
```bash
streamlit run app/app.py
```
Open the **Interactive Prediction Simulator** tab inside the Streamlit interface.

---

## 3. Supported Input Modes

### Mode 1: Manual 16-Step Entry
Enter 16 consecutive 15-minute load values ($t-15$ through $t$, in $\text{kW}$) directly in the 16-step input grid or as a comma/space-separated list. All values must be finite and non-negative.

### Mode 2: Statistically Grounded Random Scenario Generator
Generates reproducible 16-step load profiles sampled from empirical regime distributions in the selected district's training partition (`src/simulator.py:generate_scenario`):
- `normal` — **Normal Consumption:** Sampled near the median historical regime (35th–65th percentile mean load, moderate volatility).
- `high_demand` — **High Demand:** Sampled from upper-decile feeder loading windows ($\ge 85\text{th}$ percentile mean load).
- `low_demand` — **Low Demand:** Sampled from light-load night/off-peak windows ($\le 15\text{th}$ percentile mean load).
- `morning_peak` — **Morning Peak:** Sampled from empirical morning ramp windows (`06:00–10:00` local time) with positive net ramp.
- `evening_peak` — **Evening Peak:** Sampled from empirical evening peak windows (`18:00–22:00` local time).
- `irregular` — **Irregular Consumption:** Sampled from high-volatility feeder windows ($\ge 85\text{th}$ percentile intra-window standard deviation) with bounded stochastic perturbations.

### Mode 3: CSV Upload
Upload a CSV file containing at least 16 rows. When more than 16 rows are present, the simulator extracts the last 16 rows in chronological order.
- Recognized load column names: `consumption_kw`, `demand_kw`, `load_kw`, `kw`, `aggregate_demand_kw_primary`, `demand_target`, `value`, `consumption`, `demand`, `load` (or a single numeric column).
- Optional `timestamp` column: if provided, timestamps are parsed, checked for duplicate entries, and sorted chronologically.

---

## 4. Programmatic Python Usage (`src/simulator.py`)

```python
from src.simulator import PredictionSimulator

sim = PredictionSimulator()

# 1. Generate a reproducible scenario for Mathura
scenario = sim.generate_scenario(district="mathura", scenario="evening_peak", seed=42)
values_kw = scenario["consumption_kw"]

# 2. Run single-model inference
result = sim.predict_next_interval(
    scale="aggregate",
    district="mathura",
    arch="LSTM",
    feature_config="consumption_only",
    consumption_kw=values_kw,
)
print(f"Predicted t+1: {result['predicted_demand_kw']:.4f} kW")
print(f"Persistence:   {result['naive_persistence_kw']:.4f} kW")

# 3. Compare all 3 architectures (1D_CNN, LSTM, CNN_LSTM) side-by-side
for arch in ("1D_CNN", "LSTM", "CNN_LSTM"):
    res = sim.predict_next_interval(
        scale="aggregate",
        district="mathura",
        arch=arch,
        feature_config="consumption_only",
        consumption_kw=values_kw,
    )
    print(f"{arch:8s}: {res['predicted_demand_kw']:.4f} kW (delta vs persistence: {res['delta_vs_persistence_kw']:+.4f} kW)")
```
