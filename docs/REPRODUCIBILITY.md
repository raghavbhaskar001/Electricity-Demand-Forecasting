# Reproducibility Guide

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala

---

## 1. Environment Setup

### Python & Dependencies
Python 3.11+ is recommended (`3.11` – `3.14` verified). From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

- **CPU Execution:** All unit tests, aggregate baseline benchmarks, Diebold–Mariano statistical tests, interactive simulator predictions, and dashboard builds run on CPU in under 10 seconds.
- **GPU Execution:** Training all 18 individual household panel experiments (`1.07M` training windows in Mathura, `1.87M` training windows in Bareilly) requires a CUDA GPU (executed on Google Colab T4/L4 GPU via `notebooks/02_colab_training_pipeline.ipynb`).

---

## 2. Deterministic Seeding

All training and evaluation routines invoke `set_global_seed(42)` (`src/train_runner.py`), which seeds:
- Python `random.seed(42)`
- NumPy `np.random.seed(42)`
- PyTorch `torch.manual_seed(42)` and `torch.cuda.manual_seed_all(42)`
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`

---

## 3. Step-by-Step Reproduction Commands

### Step 1: Verify Unit Tests & Pipeline Integrity (No Raw Data Rebuild Needed)
Because `data/processed/mathura_features.parquet`, `data/processed/bareilly_features.parquet`, and all 18 aggregate `.pt` checkpoints (`artifacts/checkpoints/agg_*.pt`) are tracked in the repository, you can immediately run the 44-test verification suite:

```bash
python -m unittest discover tests -v
```

### Step 2: Recompute Aggregate Diebold–Mariano Statistical Tests
Re-runs the 24 pairwise Diebold–Mariano tests (with Harvey–Leybourne–Newbold adjustment and Holm–Bonferroni multiple-testing correction) directly from `artifacts/metrics/aggregate_predictions.parquet`:

```bash
python scripts/run_dm_tests.py
```

### Step 3: Rebuild Dashboard & Simulator Static Bundles
Recompiles `app/dashboard/data.js` and `app/dashboard/simulator_bundle.js` from `artifacts/metrics/*` and `artifacts/checkpoints/agg_*.pt`:

```bash
python app/build_dashboard_data.py
python app/build_simulator_bundle.py
```

### Step 4: Rebuild Processed Parquet Files from Raw Data (Optional Full Pipeline)
If you download the raw CEEW CSV archives into `data/raw/smart_meter/`:

```bash
python scripts/download_weather.py
python scripts/process_smart_meter.py
python scripts/align_weather_and_load.py
python scripts/run_baselines.py
```

### Step 5: Retrain the 36-Experiment Matrix (Optional)
- **Aggregate cohort only (`18` experiments, ~3–5 min on CPU or <1 min on GPU):**
  ```python
  from src.train_runner import run_all_experiments
  run_all_experiments(run_aggregates=True, run_individuals=False)
  ```
- **Full 36-experiment matrix (Aggregate + Individual panels on Colab GPU):**
  Generate the Colab transfer bundle with `python scripts/generate_colab_bundle.py` and run `notebooks/02_colab_training_pipeline.ipynb`.
