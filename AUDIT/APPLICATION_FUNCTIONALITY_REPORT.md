# Complete Application Functionality Audit Report

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala  
**Audit Date:** 2026-11-09  

---

## 1. Application Architecture Overview

The interactive application (`app/`) supports three execution modes using a unified codebase:
1. **Local HTTP Server + Live PyTorch API (`python app/serve_dashboard.py`):**
   - Serves `app/dashboard/` at `http://127.0.0.1:8505` (bound strictly to localhost `127.0.0.1` by default) and exposes live Python/PyTorch JSON endpoints (`GET /api/availability`, `POST /api/scenario`, `POST /api/parse_csv`, `POST /api/predict`).
2. **Streamlit Web Application (`streamlit run app/app.py`):**
   - Auto-compiles `data.js` and `simulator_bundle.js` from verified artifacts and mounts the full-bleed Midnight Grid 3D Glass interface.
3. **Direct Static Browser Execution (`app/dashboard/index.html`):**
   - Runs directly from disk (`file://`) using pre-compiled `data.js` and `simulator_bundle.js` (which embeds exact IEEE-754 `float32` tensor weights for all 18 aggregate `.pt` checkpoints).

---

## 2. Functional Audit Matrix Across All 6 Application Views

| View / Component | Audited Functionality | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **Navigation & Routing** | Sidebar navigation (`Overview`, `Forecast Analysis`, `Prediction Simulator`, `Model Comparison`, `Experiments`, `Methodology`), hash URL syncing (`#overview` … `#simulator`), responsive `ResizeObserver` chart redrawing. | Headless Node DOM harness (`verify_ui.js`, `verify_simulator_e2e.js`) | **Verified Working** |
| **1. Overview (`#view-overview`)** | 5 KPI tiles (`36 / 36` trained experiments, `3` architectures, `2` districts, `2` scales, `630,755` total test observations), 4 best-model cards, 36-cell Persistence Skill heatmap, and verified Diebold–Mariano / descriptive findings. | DOM render + data reconciliation against `training_results.json` | **Verified Working** |
| **2. Forecast Analysis (`#view-forecast`)** | **Aggregate:** Interactive SVG time-series (`#00C8E8` actual, `#F5B942` predicted, dashed persistence overlay), `1d`/`3d`/`7d`/`All` window slider, residual bar chart, error histogram, and training-log reconciliation badge (`18/18` matched within `≤ 0.0001 kW`).<br>**Individual:** Explicit disclosure notice that pointwise predictions are pending Colab export; renders per-meter sorted bar charts (`38` Mathura / `46` Bareilly meters). | DOM interaction harness (`verify_ui_interactions.js`) | **Verified Working** |
| **3. Prediction Simulator (`#view-simulator`)** | Scale / District / Architecture / Feature selectors; **Manual Entry** (16 interval inputs + quick-paste); **Random Scenario Generator** (6 training-derived presets + reproducible seed); **CSV Upload** (file picker + valid 16-row CSV template generator + `row_offset` selector); **Weather Panel** (24 pipeline test windows + paired scenario weather + editable $16 \times 9$ table); OOD warnings without clipping; 17-point SVG forecast chart; 36-combination availability matrix. | Python `test_simulator.py` (8 test methods) + Node E2E harness (`72/72` predictions matched) | **Verified Working** |
| **4. Model Comparison (`#view-compare`)** | Grouped SVG bar charts across `RMSE`, `MAE`, `WAPE`, `R²`, and `Persistence Skill` with dashed Naive Persistence baseline; highlighted metric table; Diebold–Mariano (`K = 8`, HLN + Holm-adjusted) hypothesis table for aggregate models and descriptive-only notice for individual models. | DOM render + `dm_test_results.csv` cross-check | **Verified Working** |
| **5. Experiment Explorer (`#view-experiments`)** | Real-time text search, scale/district filtering, and column-header sorting across all 36 experiment records with validation status and local artifact (`ckpt` / `preds`) indicators. | DOM interaction harness | **Verified Working** |
| **6. Methodology (`#view-method`)** | 9 structured reference cards covering datasets & DOIs, `5 × 3-min → 15-min` target construction, sequence census, feature sets, exact trainable parameter counts (`CNN1DForecaster`, `LSTMForecaster`, `CNNLSTMForecaster`), training hyperparameters, regression metric definitions, DM/HAC statistical methodology, and SHA-256 artifact hashes. | DOM render + `build_dashboard_data.py` | **Verified Working** |

---

## 3. Design System & Responsive Layout Verification

- **Theme (`MIDNIGHT GRID — 3D GLASS`):**
  - Verified palette tokens in `app/dashboard/styles.css`: `--bg-0: #071321`, `--bg-1: #0B1B2D`, `--glass: rgba(255,255,255,0.055)`, `--glass-border: rgba(255,255,255,0.12)`, `--cyan: #00C8E8`, `--blue: #3987FF`, `--amber: #F5B942`, `--ok: #10B981`, `--err: #EF4444`.
  - Controlled 3D depth on `.kpi`, `.best`, and `.sim-hero` cards with `@media (prefers-reduced-motion: reduce)` fallback.
  - Responsive breakpoints at `1280px`, `960px` (collapses sidebar to top navigation bar and switches `.sim-grid-16` to 2 columns), and `600px`.
