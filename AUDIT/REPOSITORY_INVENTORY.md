# Repository Inventory and Structural Audit

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala  
**Audit Date:** 2026-11-09  

---

## 1. Executive Summary

This document inventories the complete working tree of the repository prior to cleanup and public release. All files and directories were inspected directly on disk to determine their role, active code references, byte sizes, sensitivity, and suitability for inclusion in the public GitHub repository.

| Category | Directory / Scope | File Count | Total Size | Public Repo Recommendation |
| :--- | :--- | ---: | ---: | :--- |
| **Core ML & Inference Source** | `src/` | 22 `.py` files | 0.24 MB | **Include (Required)** |
| **Dashboard & Simulator App** | `app/` | 11 files | 7.07 MB | **Include (Required)** |
| **Evaluation & Pipeline Scripts** | `scripts/` | 16 `.py` files | 0.11 MB | **Include 10 active scripts; remove 6 one-off notebook patchers** |
| **Automated Test Suite** | `tests/` | 8 files | 0.04 MB | **Include (Required)** |
| **Configuration** | `config/` | 1 file | 0.01 MB | **Include (Required)** |
| **Trained Checkpoints & Metrics** | `artifacts/` | 58 files | 21.76 MB | **Include metrics + 18 finalized `agg_*.pt`; exclude partial `ind_*.pt`** |
| **Raw & Processed Datasets** | `data/` | 27 files | 1,050.30 MB | **Exclude raw/large parquets via `.gitignore`; document acquisition in `data/README.md`** |
| **Notebooks** | `notebooks/` | 3 files | 0.31 MB | **Include (Required)** |
| **Technical Documentation** | `docs/` | 55 files | 0.62 MB | **Include (consolidate & sanitize local file URIs)** |
| **Local Virtual Environment** | `.venv/` | 10,261 files | 94.99 MB | **Exclude (`.gitignore`)** |
| **Root-Level Duplicates & Slides** | Root directory | 10 files | 1.92 MB | **Remove exact duplicates; keep only public root files** |

---

## 2. Actual Project Structure (Pre-Cleanup)

```text
electricity-demand-forecasting/
├── .gitignore                                         (2,624 B)
├── README.md                                          (3,493 B)
├── requirements.txt                                   (1,871 B)
├── 02_colab_training_pipeline.ipynb                   (309,578 B)  [Root duplicate — older copy]
├── training_results.json                              (306,858 B)  [Root duplicate of artifacts/metrics/training_results.json]
├── master_experiment_comparison.png                   (498,618 B)  [Root duplicate of artifacts/metrics/master_experiment_comparison.png]
├── build_presentation.py                              (38,882 B)   [Internal slide-generation script]
├── FINAL_ELECTRICITY_DEMAND_FORECASTING_PRESENTATION.pptx (736,740 B) [Presentation deck]
├── FINAL_PRESENTATION_CONTENT_AND_SOURCES.md          (14,917 B)   [Internal presentation notes]
├── FINAL_PROJECT_RESULTS_AUDIT.md                     (20,931 B)   [Internal presentation audit]
├── MASTER_PRESENTATION_GENERATION_PROMPT.md           (30,283 B)   [Internal prompt file]
│
├── app/                                               (7.07 MB)
│   ├── app.py                                         (3,570 B)    [Streamlit application entry point]
│   ├── serve_dashboard.py                             (5,991 B)    [Local HTTP server + PyTorch Simulator JSON API]
│   ├── build_dashboard_data.py                        (13,478 B)   [Compiler for app/dashboard/data.js]
│   ├── build_simulator_bundle.py                      (6,894 B)    [Compiler for app/dashboard/simulator_bundle.js]
│   └── dashboard/
│       ├── index.html                                 (8,920 B)    [6-view Midnight Grid 3D Glass UI]
│       ├── styles.css                                 (23,659 B)   [Design system stylesheet]
│       ├── app.js                                     (52,001 B)   [Dashboard charts, tables, DM views]
│       ├── simulator.js                               (61,261 B)   [Interactive prediction simulator & Float32 engine]
│       ├── data.js                                    (380,395 B)  [Compiled experiment & test-series bundle]
│       └── simulator_bundle.js                        (6,817,811 B)[Compiled scalers, regime pools & 18 checkpoint tensors]
│
├── src/                                               (0.24 MB excluding __pycache__)
│   ├── __init__.py
│   ├── controlled_downloader.py                       (16,958 B)   [Dataverse streaming downloader with MD5/SHA-256 checks]
│   ├── acquire_weather.py                             (13,783 B)   [Open-Meteo ERA5-Land weather fetcher & causal aligner]
│   ├── audit_raw_data.py                              (24,776 B)   [Raw 3-minute smart-meter cadence & anomaly auditor]
│   ├── construct_targets.py                           (20,961 B)   [5×3-min → 15-min kW target builder & weather joiner]
│   ├── generate_eda.py                                (12,299 B)   [Exploratory data analysis figure generator]
│   ├── dataset.py                                     (7,422 B)    [FEATURE_SETS, contiguous windowing, 70/15/15 purge split, StandardScaler]
│   ├── audit_boundary_purge.py                        (6,649 B)    [Split boundary leakage verifier]
│   ├── reconcile_and_audit.py                         (7,023 B)    [Meter-wise sequence census reconciler]
│   ├── generate_pre_training_audit_artifacts.py       (18,982 B)   [Pre-training readiness auditor]
│   ├── generate_acquisition_report.py                 (3,419 B)    [Dataset acquisition manifest formatter]
│   ├── baselines.py                                   (13,899 B)   [Naive and seasonal (lag-96) persistence baselines]
│   ├── evaluate.py                                    (4,448 B)    [RMSE, MAE, WAPE, MAPE, R², nRMSE, Persistence Skill, Peak MAE]
│   ├── train.py                                       (3,957 B)    [CLI wrapper for master training matrix]
│   ├── train_runner.py                                (25,945 B)   [PyTorch training loop, HuberLoss, AdamW, early stopping]
│   ├── statistical_tests.py                           (6,232 B)    [Diebold–Mariano test, Newey–West HAC variance, Holm correction]
│   ├── panel_statistical_tests.py                     (6,485 B)    [Wilcoxon signed-rank & meter-blocked bootstrap tests]
│   ├── simulator.py                                   (43,254 B)   [Interactive 16-step inference, validation, CSV & scenario engine]
│   └── models/
│       ├── __init__.py                                (337 B)
│       ├── cnn1d.py                                   (2,484 B)    [CNN1DForecaster]
│       ├── lstm.py                                    (2,006 B)    [LSTMForecaster]
│       └── cnn_lstm.py                                (2,622 B)    [CNNLSTMForecaster]
│
├── scripts/                                           (0.11 MB excluding __pycache__)
│   ├── generate_aggregate_predictions.py              (5,637 B)    [Generates aggregate_test_predictions.parquet/.csv]
│   ├── generate_individual_predictions.py             (7,991 B)    [Colab/local script for individual pointwise predictions]
│   ├── run_diebold_mariano_evaluation.py              (11,233 B)   [Runs aggregate DM tests & generates dm_test_results.csv]
│   ├── test_hac_sensitivity.py                        (3,630 B)    [Runs HAC lag sensitivity K ∈ {1,2,4,8,12,16}]
│   ├── run_individual_panel_evaluation.py             (8,831 B)    [Runs individual panel Wilcoxon/bootstrap evaluation]
│   ├── generate_phase_3_4_consolidated_results.py     (2,271 B)    [Builds phase_3_4_consolidated_results.csv]
│   ├── verify_metrics_reconciliation.py               (1,777 B)    [Reconciles saved predictions vs logged metrics]
│   ├── verify_imported_checkpoints.py                 (5,040 B)    [Verifies checkpoint SHA-256 and state_dict integrity]
│   ├── package_colab_checkpoints.py                   (3,830 B)    [Colab export helper for individual checkpoints]
│   ├── run_compute_pilot.py                           (7,433 B)    [CPU vs GPU throughput benchmark script]
│   ├── apply_robust_section7.py                       (12,260 B)   [One-off notebook patching script — redundant]
│   ├── apply_section_7.py                             (10,261 B)   [One-off notebook patching script — redundant]
│   ├── setup_notebook_section7.py                     (9,337 B)    [One-off notebook patching script — redundant]
│   ├── update_colab_nb.py                             (2,571 B)    [One-off notebook patching script — redundant]
│   ├── update_notebook_export_cell.py                 (1,814 B)    [One-off notebook patching script — redundant]
│   └── verify_notebook_section7.py                    (4,808 B)    [One-off notebook verification script — redundant]
│
├── config/
│   └── config.yaml                                    (7,401 B)    [Project configuration schema]
│
├── artifacts/                                         (21.76 MB)
│   ├── checkpoints/
│   │   ├── agg_bareilly_{1D_CNN,LSTM,CNN_LSTM}_{...}.pt (9 files, 6.59 MB)  [Finalized Bareilly aggregate checkpoints]
│   │   ├── agg_mathura_{1D_CNN,LSTM,CNN_LSTM}_{...}.pt  (9 files, 6.59 MB)  [Finalized Mathura aggregate checkpoints]
│   │   └── ind_mathura_1D_CNN_consumption_only.pt     (652,844 B)  [Interrupted epoch-3 partial checkpoint]
│   └── metrics/
│       ├── training_results.json                      (306,858 B)  [Canonical 36-experiment training & test benchmark log]
│       ├── experiment_summary.csv                     (7,075 B)    [Tabular summary of all 36 experiments]
│       ├── phase_3_4_consolidated_results.csv         (8,084 B)    [36 experiments with statistical validation status]
│       ├── baseline_benchmarks.json                   (7,055 B)    [Naive & seasonal persistence benchmarks + sequence census]
│       ├── aggregate_test_predictions.parquet         (213,132 B)  [23,733 pointwise aggregate test predictions]
│       ├── aggregate_test_predictions.csv             (3,554,216 B)[CSV export of aggregate_test_predictions.parquet]
│       ├── dm_test_results.csv                        (5,108 B)    [16 aggregate Diebold–Mariano hypothesis test results]
│       ├── dm_test_summary.json                       (12,453 B)   [Detailed JSON output of DM evaluation]
│       ├── hac_lag_sensitivity.csv                    (2,067 B)    [Newey–West bandwidth sensitivity table]
│       ├── checkpoint_integrity_manifest.csv          (6,842 B)    [Inventory of 19 local .pt files]
│       ├── compute_pilot_results.json                 (868 B)      [Timing pilot output]
│       ├── pre_training_audit_summary.json            (3,221 B)    [Pre-training data integrity summary]
│       ├── dm_significance_comparison.png             (328,320 B)  [DM test plot]
│       ├── master_experiment_comparison.png           (498,618 B)  [Master RMSE/Skill comparison plot]
│       ├── eda/fig1..fig5_*.png                       (5 files, 3.98 MB) [EDA plots]
│       └── training_logs/agg_*.json                   (18 files, 39.4 KB) [Epoch-by-epoch train/val loss curves]
│
├── data/                                              (1,050.30 MB)
│   ├── raw/
│   │   ├── ceew/bareilly/                             (3 files, 661.54 MB) [Harvard Dataverse doi:10.7910/DVN/GOCHJH]
│   │   ├── ceew/mathura/                              (3 files, 385.38 MB) [Harvard Dataverse doi:10.7910/DVN/GOCHJH]
│   │   ├── ceew/CEEW -  ReadMe.txt                    (4,436 B)
│   │   ├── iiit_delhi/                                (2 files, 30.68 MB)  [Harvard Dataverse doi:10.7910/DVN/7MEXN4]
│   │   └── weather/                                   (6 files, 9.83 MB)   [Open-Meteo ERA5-Land hourly CSV/JSON]
│   ├── interim/                                       (.gitkeep)
│   └── processed/                                     (9 files, 13.88 MB)
│       ├── mathura_aggregate.parquet                  (611,750 B)
│       ├── mathura_features.parquet                   (1,241,610 B)
│       ├── mathura_individual_meters.parquet          (2,450,770 B)
│       ├── bareilly_aggregate.parquet                 (831,309 B)
│       ├── bareilly_features.parquet                  (1,688,804 B)
│       ├── bareilly_individual_meters.parquet         (3,883,486 B)
│       ├── delhi_aggregate.parquet                    (277,748 B)
│       ├── delhi_features.parquet                     (581,074 B)
│       └── delhi_individual_meters.parquet            (2,310,801 B)
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb             (11,675 B)
│   └── 02_colab_training_pipeline.ipynb               (311,751 B)
│
├── tests/                                             (8 files, 41.0 KB)
│   ├── test_baselines_and_metrics.py
│   ├── test_data_integrity.py
│   ├── test_dataset_slicing.py
│   ├── test_panel_statistical_tests.py
│   ├── test_simulator.py
│   ├── test_statistical_tests.py
│   └── test_target_construction_phase1_5.py
│
└── docs/                                              (55 files, 0.62 MB)
    ├── audit_results/                                 (8 files)
    ├── dataset_inventory/                             (16 files)
    └── PHASE_*.md & audit CSVs                        (31 files)
```

---

## 3. Directory-by-Directory Assessment

| Directory | Purpose | Required for App / Reproducibility? | Referenced by Active Code? | Include in Public Repo? | Contains Sensitive or Large Material? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `src/` | Core data ingestion, target construction, sequence splitting, PyTorch models (`1D_CNN`, `LSTM`, `CNN_LSTM`), training runner, statistical tests, and live simulator engine. | **Yes** | **Yes** (`app/`, `scripts/`, `tests/`, `notebooks/`) | **Yes** | No (0.24 MB source code; `__pycache__` excluded). |
| `app/` | Streamlit entry point, local HTTP + PyTorch API server, bundle compilers, and Midnight Grid 3D Glass frontend (`index.html`, `styles.css`, `app.js`, `simulator.js`, `data.js`, `simulator_bundle.js`). | **Yes** | **Yes** | **Yes** | No secrets. `simulator_bundle.js` is 6.50 MB (contains float32 weights of the 18 aggregate checkpoints for zero-server browser execution). |
| `scripts/` | CLI scripts for prediction regeneration, Diebold–Mariano tests, HAC sensitivity, and checkpoint verification. Also contains 6 one-off notebook-patching scripts. | **Yes** (10 core scripts); **No** (6 one-off notebook patchers) | **Yes** (referenced in docs and notebooks) | **Include 10 core scripts; remove 6 redundant patchers** | `scripts/package_colab_checkpoints.py` contains a comment with `<repo-root>` (to be sanitized). |
| `artifacts/checkpoints/` | Holds 18 finalized aggregate `.pt` checkpoints (`agg_*.pt`, 13.18 MB total) and 1 interrupted partial individual checkpoint (`ind_mathura_1D_CNN_consumption_only.pt`, 0.65 MB). | **Yes** (18 `agg_*.pt` required by `src/simulator.py` and `scripts/generate_aggregate_predictions.py`) | **Yes** | **Include 18 finalized `agg_*.pt`; exclude partial `ind_*.pt`** | Total size of 18 `agg_*.pt` is 13.18 MB (~650–870 KB each), well within GitHub limits. |
| `artifacts/metrics/` | Canonical experimental records (`training_results.json`, `baseline_benchmarks.json`, `dm_test_results.csv`, `hac_lag_sensitivity.csv`, `aggregate_test_predictions.parquet`, training logs, and plots). | **Yes** | **Yes** (`app/build_dashboard_data.py`, `src/simulator.py`, `tests/`) | **Yes** (consider excluding redundant 3.55 MB `.csv` copy of `aggregate_test_predictions.parquet` or keeping only `.parquet`) | No sensitive data. |
| `data/raw/` | Untouched CEEW smart-meter CSV/TAB files (1.02 GB), IIIT Delhi `.tab`/`.pdf` (30.7 MB), and Open-Meteo weather downloads (9.8 MB). | Needed only when rebuilding `data/processed/` from scratch | Read by `src/audit_raw_data.py` and `src/construct_targets.py` | **No** (excluded via `.gitignore`; provide `data/README.md` with official Dataverse DOIs & downloader commands) | Very large (**1.04 GB**). Publicly licensed (CC0 1.0 / CC BY 4.0), but far too large for Git hosting. |
| `data/processed/` | Cleaned 15-minute aggregate and individual parquet files (13.88 MB total). Wait: `mathura_features.parquet` (1.24 MB) and `bareilly_features.parquet` (1.69 MB) are read by `src/simulator.py` and `tests/`. | `mathura_features.parquet` & `bareilly_features.parquet` (2.93 MB combined) are required by `src/simulator.py` and `tests/` | **Yes** (`src/simulator.py`, `tests/test_dataset_slicing.py`, `tests/test_baselines_and_metrics.py`, `tests/test_simulator.py`) | **Include `mathura_features.parquet` & `bareilly_features.parquet` (2.93 MB)** so `pytest`/`unittest` and `serve_dashboard.py` work out-of-the-box; ignore larger individual meter parquets (`data/processed/*_individual_meters.parquet`). | Aggregated feeder parquets (`*_features.parquet`) contain only summed cohort load and Open-Meteo weather (zero household-level identifiers). |
| `config/` | `config/config.yaml` experimental configuration. | **Yes** | Referenced in documentation | **Yes** (update to match exact `train_runner.py` hyperparameters) | No (7.4 KB). |
| `notebooks/` | `01_exploratory_data_analysis.ipynb` and `02_colab_training_pipeline.ipynb`. | **Yes** | Referenced in docs | **Yes** (sanitize local path strings in cell outputs) | No secrets (0.31 MB). |
| `tests/` | 7 Python `unittest` modules (44 tests covering data integrity, target formulas, splitting, baselines, DM/panel statistics, and live simulator). | **Yes** | Executed by `unittest` / `pytest` | **Yes** | No (41 KB). |
| `docs/` | Phase audit reports, dataset manifests, and methodology documentation. | **Yes** | Referenced across reports | **Yes** (add structured user-facing docs `METHODOLOGY.md`, `DATASETS.md`, `MODEL_ARCHITECTURES.md`, `INFERENCE_GUIDE.md`, `REPRODUCIBILITY.md`, `LIMITATIONS.md`, `REFERENCES.md` and sanitize `file:///e:/...` URIs) | Contains `...` local workspace links in 12 markdown files (to be converted to relative paths). |
| `.venv/` | Local Python virtual environment. | Local only | No | **No** (excluded via `.gitignore`) | 94.99 MB local binaries. |

---

## 4. Duplicate, Redundant, and Temporary Files Identified

1. **Exact Root-Level Duplicates (Identical SHA-256 to `artifacts/metrics/`):**
   - `training_results.json` (root, `306,858 B`) == `artifacts/metrics/training_results.json` (`306,858 B`).
   - `master_experiment_comparison.png` (root, `498,618 B`) == `artifacts/metrics/master_experiment_comparison.png` (`498,618 B`).
2. **Stale Root-Level Notebook Copy:**
   - `02_colab_training_pipeline.ipynb` (root, `309,578 B`) is an older copy of `notebooks/02_colab_training_pipeline.ipynb` (`311,751 B`).
3. **One-Off Notebook Patching Scripts in `scripts/`:**
   - `scripts/apply_robust_section7.py`, `scripts/apply_section_7.py`, `scripts/setup_notebook_section7.py`, `scripts/update_colab_nb.py`, `scripts/update_notebook_export_cell.py`, and `scripts/verify_notebook_section7.py` were temporary scripts used to insert cells into `notebooks/02_colab_training_pipeline.ipynb` and are not part of the ML or evaluation pipeline.
4. **Internal Presentation Generation Files in Root / `docs/`:**
   - `build_presentation.py`, `FINAL_ELECTRICITY_DEMAND_FORECASTING_PRESENTATION.pptx`, `FINAL_PRESENTATION_CONTENT_AND_SOURCES.md`, `FINAL_PROJECT_RESULTS_AUDIT.md`, `MASTER_PRESENTATION_GENERATION_PROMPT.md`, and `docs/FINAL_PRESENTATION_MASTER_PROMPT.md`.
5. **Bytecode Cache Directories (`__pycache__/`):**
   - Present in `src/__pycache__/`, `src/models/__pycache__/`, `scripts/__pycache__/`, `app/__pycache__/`, and `tests/__pycache__/` (already ignored by `.gitignore`, can be cleaned locally).
