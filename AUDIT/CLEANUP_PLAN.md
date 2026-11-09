# Phase 6 — Repository Cleanup & Restructuring Plan

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala

---

## 1. Safety Policy

Per Phase 6 directives:
- **No potentially useful research artifacts or files have been deleted automatically.**
- All candidate deletions are itemized below with their exact byte sizes, SHA-256 or structural relationship to canonical files, and reason for removal.
- Until explicit user confirmation is received, these files remain untouched on disk and are excluded from version control via `.gitignore`.

---

## 2. Candidate Files Proposed for Deletion (Requires User Approval)

### Category A — Exact Root-Level Duplicates of Canonical `artifacts/metrics/` Files
| File Path | Size | Canonical Counterpart | Verification |
| :--- | ---: | :--- | :--- |
| `training_results.json` | 306,858 B | `artifacts/metrics/training_results.json` | Byte-for-byte identical copy left in root during Colab download. |
| `master_experiment_comparison.png` | 498,618 B | `artifacts/metrics/master_experiment_comparison.png` | Byte-for-byte identical copy left in root during Colab download. |

### Category B — Stale Root-Level Notebook Duplicate
| File Path | Size | Canonical Counterpart | Verification |
| :--- | ---: | :--- | :--- |
| `02_colab_training_pipeline.ipynb` | 309,578 B | `notebooks/02_colab_training_pipeline.ipynb` (311,751 B) | Older draft before Section 7 checkpoint-packaging audit updates. |

### Category C — One-Off Notebook-Patching Scripts in `scripts/`
These six scripts were single-use utilities written solely to insert or verify Section 7 cells in `notebooks/02_colab_training_pipeline.ipynb` and are not part of the data, training, evaluation, or packaging pipeline:
| File Path | Size | Purpose |
| :--- | ---: | :--- |
| `scripts/apply_robust_section7.py` | 15,675 B | One-off cell injector for Section 7 of `02_colab_training_pipeline.ipynb`. |
| `scripts/apply_section_7.py` | 12,188 B | Earlier draft of Section 7 cell injector. |
| `scripts/setup_notebook_section7.py` | 14,033 B | Earlier draft of Section 7 setup utility. |
| `scripts/update_colab_nb.py` | 16,690 B | One-off notebook JSON updater. |
| `scripts/update_notebook_export_cell.py` | 3,907 B | Single-cell patch utility for notebook export cell. |
| `scripts/verify_notebook_section7.py` | 4,768 B | One-off inspection script superseded by `tests/`. |

### Category D — Internal Presentation Generation Prompts & Slide Deck Artifacts
| File Path | Size | Purpose |
| :--- | ---: | :--- |
| `build_presentation.py` | 45,563 B | Script used to generate the 8-slide PowerPoint presentation. |
| `FINAL_ELECTRICITY_DEMAND_FORECASTING_PRESENTATION.pptx` | 666,227 B | Generated 8-slide PowerPoint deck (optional keep if user wants slides in repo, or move to `docs/slides/`). |
| `FINAL_PRESENTATION_CONTENT_AND_SOURCES.md` | 21,984 B | Slide-by-slide notes generated for the presentation task. |
| `FINAL_PROJECT_RESULTS_AUDIT.md` | 18,476 B | Earlier audit summary superseded by `AUDIT/` reports and `docs/`. |
| `MASTER_PRESENTATION_GENERATION_PROMPT.md` | 49,336 B | Prompt artifact generated for external presentation builders. |
| `docs/FINAL_PRESENTATION_MASTER_PROMPT.md` | 50,023 B | Duplicate prompt artifact in `docs/`. |

---

## 3. Canonical Repository Structure (Post-Cleanup Target)

```text
electricity-demand-forecasting-dl/
├── app/
│   ├── app.py                          # Streamlit interactive dashboard & simulator
│   ├── build_dashboard_data.py         # Compiles artifacts/metrics into dashboard/data.js
│   ├── build_simulator_bundle.py       # Exports Float32 weights & scalers to simulator_bundle.js
│   ├── serve_dashboard.py              # Hardened local HTTP + /api/* server
│   └── dashboard/
│       ├── index.html                  # 7-view research dashboard & simulator UI
│       ├── styles.css                  # Midnight Grid glassmorphism stylesheet
│       ├── app.js                      # Dashboard visualization controller
│       ├── simulator.js                # Interactive simulator & client-side neural engine
│       ├── data.js                     # Compiled empirical benchmark & DM test data
│       └── simulator_bundle.js         # Compiled Float32 weights, scalers & historical windows
├── artifacts/
│   ├── checkpoints/
│   │   └── agg_*.pt                    # 18 verified aggregate PyTorch checkpoints (4.18 MB)
│   └── metrics/
│       ├── aggregate_predictions.parquet
│       ├── baseline_benchmarks.json
│       ├── dm_statistical_tests.csv
│       ├── dm_statistical_tests.json
│       ├── master_experiment_comparison.png
│       └── training_results.json
├── AUDIT/
│   ├── REPOSITORY_INVENTORY.md
│   ├── RESEARCH_COMPLETENESS_REPORT.md
│   ├── INFERENCE_VERIFICATION_REPORT.md
│   ├── APPLICATION_FUNCTIONALITY_REPORT.md
│   ├── SECURITY_AUDIT_REPORT.md
│   ├── CLEANUP_PLAN.md
│   └── FINAL_RELEASE_VALIDATION.md
├── config/
│   └── config.yaml                     # Experimental configuration schema
├── data/
│   ├── README.md                       # Dataset documentation, DOIs & rebuild guide
│   ├── raw/                            # Ignored in Git (.gitkeep preserved)
│   ├── interim/                        # Ignored in Git (.gitkeep preserved)
│   └── processed/
│       ├── mathura_features.parquet    # Tracked compact aggregate dataset (1.42 MB)
│       └── bareilly_features.parquet   # Tracked compact aggregate dataset (1.51 MB)
├── docs/
│   ├── METHODOLOGY.md
│   ├── DATASETS.md
│   ├── MODEL_ARCHITECTURES.md
│   ├── INFERENCE_GUIDE.md
│   ├── REPRODUCIBILITY.md
│   ├── LIMITATIONS.md
│   ├── REFERENCES.md
│   └── phase_*.md                      # Chronological phase research records
├── notebooks/
│   ├── 01_eda_and_data_audit.ipynb
│   └── 02_colab_training_pipeline.ipynb
├── scripts/
│   ├── download_weather.py
│   ├── process_smart_meter.py
│   ├── align_weather_and_load.py
│   ├── generate_colab_bundle.py
│   ├── generate_individual_predictions.py
│   ├── package_colab_checkpoints.py
│   ├── run_baselines.py
│   ├── run_dm_tests.py
│   └── verify_colab_export.py
├── src/
│   ├── __init__.py
│   ├── baselines.py
│   ├── config_loader.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── models.py
│   ├── simulator.py
│   ├── statistical_tests.py
│   └── train_runner.py
├── tests/
│   ├── test_audit_fixes.py
│   ├── test_pipeline.py
│   └── test_simulator.py
├── .env.example
├── .gitignore
├── .python-version
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```
