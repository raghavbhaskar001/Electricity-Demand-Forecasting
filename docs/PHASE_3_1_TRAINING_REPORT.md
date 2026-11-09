# Phase 3.1: Pooled Individual-Meter Training Execution Report

**Project Title:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Execution Timestamp:** 2026-11-02 13:58 IST  
**Primary Target Scope:** Pooled Individual Smart Meter Panel Models (CEEW Mathura & CEEW Bareilly)  
**Status:** Verification Certified & Phase 3.1 Training Executed  

---

## 1. Execution Scope & Cohort Boundaries

The Phase 3.1 training scope encompasses the complete **18 Pooled Individual-Meter Deep Learning Experiments**:

```
Pooled Individual Matrix: 18 Experiments
├── Mathura Individual Pooled (38 meters): 9 runs
│   ├── 1D CNN × [Consumption-Only, Retro Weather, Operational Weather]
│   ├── LSTM × [Consumption-Only, Retro Weather, Operational Weather]
│   └── CNN–LSTM × [Consumption-Only, Retro Weather, Operational Weather]
└── Bareilly Individual Pooled (46 meters): 9 runs
    ├── 1D CNN × [Consumption-Only, Retro Weather, Operational Weather]
    ├── LSTM × [Consumption-Only, Retro Weather, Operational Weather]
    └── CNN–LSTM × [Consumption-Only, Retro Weather, Operational Weather]
```

### 1.1 Non-Inclusion of IIIT Delhi
- **Census State:** 19 apartments, 262,680 contiguous sequences (Train: 183,866, Val: 39,393, Test: 38,813).
- **Scientific Rationale:** As documented in Phase 2.0 and Phase 3.0, IIIT Delhi exhibits non-synchronous staggered occupancy (only 96 contiguous 16-step windows where all 19 apartments reported simultaneously). Furthermore, Delhi lacks the localized rural feeder context and operational 5-day weather publication lag inquiry central to CEEW Mathura and Bareilly.
- **Boundary Guarantee:** The 18 individual deep learning models cover **Mathura and Bareilly** smart meters exclusively. Delhi is retained strictly for baseline descriptive analysis.

---

## 2. Mandatory Pre-Execution Audits & Implementation Verification

| Verification Check | Audited Implementation | Verdict |
| :--- | :--- | :---: |
| **1. Execution Environment** | Verified in `notebooks/02_colab_training_pipeline.ipynb` and `src/train_runner.py`. Primary GPU runtime targets NVIDIA Tesla T4 / A100 in Google Colab; local runner executes via multi-core PyTorch CPU. | **PASS** |
| **2. Target Scaler Inversion** | Slicing and inverse transform handled by `PooledMeterTargetScaler` in `src/train_runner.py`. Slices test predictions by meter index: $\hat{y}_{\text{kW}}^{(m)} = \hat{z}^{(m)} \cdot \sigma_{\text{train}}^{(m)} + \mu_{\text{train}}^{(m)}$. | **PASS** |
| **3. Scaler Fitting Isolation** | Per-meter standard scalers ($\mu_{\text{train}}^{(m)}, \sigma_{\text{train}}^{(m)}$) are fitted strictly on each meter's training split. Validation and test partitions are transformed using frozen training statistics. | **PASS** |
| **4. Physical kW Evaluation** | Test evaluation metrics (RMSE, MAE, WAPE, MAPE, $R^2$, Peak MAE) are calculated strictly against raw physical kW actuals after inverse transformation. | **PASS** |
| **5. Meter Identity Preservation** | Meter slices are tracked continuously (`(meter_id, start_idx, end_idx, scaler)`), enabling both pooled global evaluation and per-meter distribution analysis. | **PASS** |
| **6. Sample Identicality** | Fixed seed 42, 70/15/15 chronological split, and 16-step boundary purge gap guarantee that all 3 model families evaluate on identical sequence tensors. | **PASS** |
| **7. Test Partition Isolation** | Early stopping and checkpoint restoration track `evaluate_loss(model, va_loader, ...)` on validation data only. The test partition is never accessed during optimization. | **PASS** |

---

## 3. Training Protocol & Technical Hyperparameters

- **Loss Objective:** Huber Loss ($\delta = 1.0$) operating in **standardized target space** ($z \sim \mathcal{N}(0, 1)$), ensuring scale-invariant gradient dampening ($L_1$ behavior for deviations $>1\sigma$).
- **Optimization:** AdamW ($\text{LR} = 10^{-3}$, $\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-4}$), gradient clipping norm $= 1.0$.
- **Scheduler:** `ReduceLROnPlateau` monitoring validation loss (factor $0.5$, patience $4$ epochs, min LR $10^{-6}$).
- **Batch Size:** 512 (selected to maximize GPU/CPU tensor throughput).
- **Sequence Extraction:** Optimized with vectorized sliding window striding (`np.lib.stride_tricks.sliding_window_view`), reducing extraction latency across 1.54M sequences from 60 seconds to **0.36 seconds**.

---

## 4. Benchmark Baselines for Direct Model Comparison

Evaluated on identical chronological test partitions in `artifacts/metrics/baseline_benchmarks.json`:

| Cohort | Baseline Model | Test Sequences | RMSE (kW) | MAE (kW) | WAPE (%) | MAPE (%) | $R^2$ Score | $SS_{\text{pers}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mathura Individual (38 meters)** | **Pooled Naive Persistence** | 229,281 | **0.2110** | **0.0707** | **38.13%** | **54.60%** | **0.6002** | 0.0000 |
| Mathura Individual (38 meters) | Meter-Wise Median | 38 meters | 0.1558 | 0.0578 | 44.86% | — | 0.3769 | 0.0000 |
| **Bareilly Individual (46 meters)** | **Pooled Naive Persistence** | 398,897 | **0.2755** | **0.1114** | **27.22%** | **39.49%** | **0.7425** | 0.0000 |
| Bareilly Individual (46 meters) | Meter-Wise Median | 46 meters | 0.2312 | 0.0911 | 31.05% | — | 0.6214 | 0.0000 |

---

## 5. Google Colab GPU Execution Instructions

To execute or re-run the 18 pooled individual-meter models with full GPU acceleration:
1. Open [`notebooks/02_colab_training_pipeline.ipynb`](../notebooks/02_colab_training_pipeline.ipynb) in Google Colab.
2. Select Runtime $\rightarrow$ Change runtime type $\rightarrow$ **T4 GPU** (or A100).
3. Execute all cells. The pipeline automatically:
   - Detects the GPU model and available VRAM.
   - Executes the 18 aggregate models and 18 pooled individual models.
   - Saves model weights to `artifacts/checkpoints/` and logs to `artifacts/metrics/training_logs/`.
   - Generates consolidated comparison visualizations and updates `artifacts/metrics/experiment_summary.csv`.

