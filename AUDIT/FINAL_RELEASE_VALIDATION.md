# Phase 9 & Pre-Publication Gate — Final Release Validation Report

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala  
**Validation Date:** 2026-11-09

---

## 1. Executive Summary

All source code modules, static bundle compilers, statistical hypothesis scripts, interactive simulator endpoints, security controls, dependency declarations, dataset licenses, file-size limits, and Git history checks were verified end-to-end. Every pre-publication gate check passed with zero failures.

| Pre-Publication Gate Check | Scope | Result | Summary |
| :--- | :--- | :---: | :--- |
| **1. Cleanup & Preservation Policy** | 15 duplicate/one-off/presentation files in `AUDIT/CLEANUP_PLAN.md` | **PASS** | All 15 files preserved locally on disk; excluded from Git via `.gitignore` |
| **2. Git History & Secret Security Audit** | All commits, branches, tags, reflogs, Git objects, notebooks & 195 tracked files | **PASS** | `0` secrets, `0` historical commits/leaks, `0` local paths, Colab `userId` stripped |
| **3. Python & Frontend Dependency Audit** | 12 Python packages (`requirements.txt` + installed) & 6 frontend dashboard files | **PASS** | `pip check` clean; CVE-safe floor versions enforced (`torch>=2.6.0`, `requests>=2.32.4`, `streamlit>=1.40.0`); `0` external frontend CDN dependencies |
| **4. Dataset & Licensing Verification** | `LICENSE`, `CITATION.cff`, `README.md`, `data/README.md`, and tracked Parquets | **PASS** | MIT license scoped strictly to code; CEEW (CC0 1.0) & ERA5-Land (CC BY 4.0) properly attributed; raw/IIIT-D data excluded from Git |
| **5. GitHub File-Size Audit** | All `195` staged files (`32.38 MB` total) | **PASS** | Max file `6.50 MB` (`simulator_bundle.js`); `0` files $\ge 10\text{ MB}`; well below GitHub 50 MB / 100 MB limits |
| **6. Unit, API & Reproducibility Suite** | `45` Python modules, `44` unit tests, HTTP security & Float32 reconciliation | **PASS** | `44 / 44` unit tests passing in `2.26s`; max PyTorch deviation `0.00e+00 kW` |

---

## 2. Dedicated Git History & Secret Security Audit

### 2.1 Git History, Branches, Tags, Reflogs & Object Database Inspection
Executed across `.git/`:
- `git branch -a`: No prior branches containing unverified commits.
- `git tag -l`: `0` tags.
- `git rev-list --all`: `0` prior commits in history before the clean initial commit on `main`.
- `git reflog --all`: `0` prior reflog entries or orphaned commit chains.
- `git fsck --full`: `0` dangling blobs, trees, or unreachable commits containing removed credentials.
- Historical `.env` files: `0` `.env` files have ever existed in the working tree or Git object store. `.env` and `.env.*` are blocked by `.gitignore`, while `!.env.example` provides non-sensitive default host/port placeholders (`DASHBOARD_HOST=127.0.0.1`, `DASHBOARD_PORT=8505`).

### 2.2 Notebook Cell Outputs & Metadata Sanitization
- `notebooks/01_exploratory_data_analysis.ipynb` (`14` cells): `0` cell outputs, `0` execution metadata blocks, `0` local paths.
- `notebooks/02_colab_training_pipeline.ipynb` (`18` cells, `4` cells with empirical training outputs):
  - Fixed a JSON string escape on line 2127 and verified strict JSON schema compliance (`json.loads`).
  - Removed Google Colab's automatic `metadata.executionInfo` (which contained an internal Google numeric `userId`) and `metadata.colab.base_uri` across all 5 code cells that had them, while preserving all 4 empirical training log/figure outputs intact.

### 2.3 Secret & Local-Path Scan Across All 195 Staged Files
Scanned all non-binary staged files against Gitleaks-equivalent rule patterns (`aws_access_key`, `aws_secret_key`, `github_pat`, `github_fine_grained`, `google_api_key`, `google_oauth`, `slack_token`, `private_key_block`, `generic_bearer_or_secret`, `mongodb_postgres_uri`) and local-path regexes (`E:\CNN model`, `e:/CNN model`, `C:\Users\`):
- **Secret Hits:** `0` (no credential revocation or rotation required).
- **Local Machine Path Hits:** `0` (additionally replaced `BASE_DIR = Path("e:/CNN model")` with portable `BASE_DIR = Path(__file__).resolve().parent.parent` in `src/acquire_weather.py`, `src/audit_raw_data.py`, `src/construct_targets.py`, and `src/generate_eda.py`, and converted `docs/dataset_inventory/weather_manifest.csv` paths to relative `data/raw/weather/...`).

---

## 3. Dependency & Vulnerability Audit

### 3.1 Python Dependencies (`requirements.txt` vs. Installed Environment)
`python -m pip check` returned exit code `0` (`No broken requirements found.`). Unused packages (`openpyxl`, `python-dotenv`, `tqdm`, `altair`) were removed from `requirements.txt`.

| Package | Declared Floor (`requirements.txt`) | Verified Installed Version | CVE / Advisory Audit & Notes |
| :--- | :--- | :--- | :--- |
| `numpy` | `>=2.0.0` | `2.5.2` | Clean; no active CVEs affecting array operations on trusted inputs |
| `pandas` | `>=2.2.0` | `3.0.5` | Clean; no `eval`/`query` on untrusted strings used |
| `scipy` | `>=1.13.0` | `1.18.1` | Clean |
| `scikit-learn` | `>=1.5.0` | `1.9.0` | Clean; `StandardScaler` fit strictly in-memory on training partitions |
| `pyarrow` | `>=17.0.0` | `25.0.1` | Clean; `>=14.0.1` patches `CVE-2023-47248` arbitrary code execution in IPC/Parquet |
| `torch` | `>=2.6.0` | `2.15.0` | Floor raised from `>=2.2.0` to `>=2.6.0` to include upstream fix for `CVE-2025-32434` (`torch.load`); additionally guarded by SHA-256 manifest & allowlist containment in `src/simulator.py` |
| `requests` | `>=2.32.4` | `2.34.2` | Floor raised from `>=2.32.0` to `>=2.32.4` to include upstream fixes for `CVE-2024-35195` (cert verification) and `CVE-2024-47081` (`.netrc` credential leak) |
| `pyyaml` | `>=6.0.1` | `6.0.3` | Clean; codebase exclusively uses `yaml.safe_load()` (`src/config_loader.py`) |
| `matplotlib` | `>=3.9.0` | `3.11.1` | Clean |
| `seaborn` | `>=0.13.0` | `0.13.2` | Clean |
| `streamlit` | `>=1.40.0` | `1.62.0` | Floor raised from `>=1.38.0` to `>=1.40.0` (well past `CVE-2024-42474` patched in `1.37.0`) |
| `pytest` | `>=8.3.0` | `9.1.1` | Clean (dev/test runner) |

### 3.2 Frontend Dependency Audit (`app/dashboard/`)
- Inspected `app/dashboard/index.html`, `styles.css`, `app.js`, `simulator.js`, `data.js`, and `simulator_bundle.js`.
- **External URLs / CDNs:** `0` (`external_urls_in_index_html = []`).
- **Third-Party JS Libraries:** `0` (vanilla ES6 JavaScript, custom HTML5 Canvas charts, and in-repo Float32 tensor math). Zero npm supply-chain exposure.

---

## 4. Dataset Redistribution & Licensing Verification

| Dataset / Artifact | Staged in Git? | Upstream License & DOI | Redistribution Verification |
| :--- | :---: | :--- | :--- |
| **Repository Source Code, Scripts, Tests & Docs** | Yes | **MIT License** (Copyright © 2026 Raghav Bhaskar) | Original academic engineering work; explicitly scoped to software in `LICENSE`. |
| **Derived Aggregate Feature Tables** (`data/processed/mathura_features.parquet`, `bareilly_features.parquet`, `2.79 MB` total) | Yes | **CEEW Smart Meter Data:** CC0 1.0 Public Domain (`doi:10.7910/DVN/N3615P`)<br>**ERA5-Land Weather:** Copernicus C3S / CC BY 4.0 (`doi:10.24381/cds.e2161bac`) | **Permitted.** CC0 1.0 places smart-meter telemetry in the public domain without restriction; Copernicus C3S / CC BY 4.0 permits redistribution of adapted/resampled weather features with attribution (provided in `LICENSE`, `CITATION.cff`, `README.md`, and `data/README.md`). |
| **Raw CEEW CSV Archives** (`data/raw/ceew/`, `~1.05 GB`) & **Individual Household Parquets** (`*_individual_meters.parquet`, `318.93 MB`) | **No** (Ignored via `.gitignore`) | CC0 1.0 (`doi:10.7910/DVN/N3615P`) | Excluded to keep the Git repository lightweight (`32.38 MB`); reproducible via `scripts/` from the upstream Harvard Dataverse DOI. |
| **IIIT Delhi Raw & Processed Data** (`data/raw/iiit_delhi/`, `data/processed/delhi_*.parquet`) | **No** (Ignored via `.gitignore`) | Academic research license (`doi:10.6084/m9.figshare.c.4490387`) | Excluded from redistribution; only aggregate descriptive census statistics (`docs/audit_results/iiit_delhi_*.csv`) and upstream metadata citation are included. |
| **Trained Aggregate Model Weights** (`artifacts/checkpoints/agg_*.pt`, `18` files, `12.58 MB` total) | Yes | **MIT License** | Learned neural network parameters produced by this project's training pipeline. |

---

## 5. GitHub File-Size Audit

### 5.1 Overall Repository Size Summary
- **Total Staged Files:** `195` files
- **Total Uncompressed Size:** `33,955,153 bytes` (**`32.382 MB`**)
- **Files $\ge 100\text{ MB}$ (GitHub Hard Limit):** `0`
- **Files $\ge 50\text{ MB}$ (GitHub Warning Threshold):** `0`
- **Files $\ge 10\text{ MB}$:** `0`
- **Largest Single File:** `app/dashboard/simulator_bundle.js` (**`6.502 MB`**, `6,817,793 bytes`)

### 5.2 Breakdown by File Extension
| Extension | File Count | Total Size (Bytes) | Total Size (MB) | Purpose |
| :--- | ---: | ---: | ---: | :--- |
| `.pt` | 18 | 13,189,724 | 12.579 MB | Verified PyTorch aggregate checkpoints (`agg_*.pt`, `0.62–0.83 MB` each) |
| `.js` | 4 | 7,312,234 | 6.973 MB | Dashboard UI (`app.js`, `simulator.js`) & compiled data bundles (`data.js`, `simulator_bundle.js`) |
| `.png` | 7 | 4,806,499 | 4.584 MB | Master comparison plot, DM significance chart, and 5 EDA figures |
| `.csv` | 24 | 3,662,792 | 3.493 MB | Aggregate test predictions (`3.39 MB`), DM test tables, and data audit summaries |
| `.parquet` | 3 | 3,143,546 | 2.998 MB | `bareilly_features.parquet` (`1.61 MB`), `mathura_features.parquet` (`1.18 MB`), `aggregate_test_predictions.parquet` (`0.20 MB`) |
| `.json` | 27 | 463,574 | 0.442 MB | `training_results.json`, `baseline_benchmarks.json`, `dm_test_summary.json`, and per-run training logs |
| `.md` | 40 | 419,800 | 0.400 MB | Root documentation, `docs/` technical guides, phase reports, and `AUDIT/` reports |
| `.py` | 43 | 375,455 | 0.358 MB | Core package (`src/`), pipeline scripts (`scripts/`), apps (`app/`), and test suite (`tests/`) |
| `.ipynb` | 2 | 324,510 | 0.309 MB | `01_exploratory_data_analysis.ipynb` (`11.4 KB`) & `02_colab_training_pipeline.ipynb` (`305.5 KB`) |
| `.pdf` | 1 | 189,916 | 0.181 MB | Upstream dataset documentation (`docs/dataset_inventory/metadata_cache/dataset_a_desc.pdf`) |
| Other (`.txt`, `.css`, `.html`, `.yaml`, `.cff`, `.example`, `<no_ext>`) | 26 | 67,103 | 0.064 MB | Config, stylesheets, HTML shell, license, citation, `.gitignore`, `.gitkeep` |
| **Total** | **195** | **33,955,153** | **32.382 MB** | — |

### 5.3 All Files Exceeding 1 MB (`6` files)
1. `app/dashboard/simulator_bundle.js` — `6.502 MB` (`6,817,793 B`)
2. `artifacts/metrics/aggregate_test_predictions.csv` — `3.390 MB` (`3,554,216 B`)
3. `data/processed/bareilly_features.parquet` — `1.611 MB` (`1,688,804 B`)
4. `artifacts/metrics/eda/fig4_weather_demand_relationships.png` — `1.585 MB` (`1,661,877 B`)
5. `artifacts/metrics/eda/fig3_household_load_diversity.png` — `1.544 MB` (`1,619,382 B`)
6. `data/processed/mathura_features.parquet` — `1.184 MB` (`1,241,610 B`)

### 5.4 Git LFS & Release-Asset Strategy Recommendation
- **Current Repository (`32.38 MB` total, max file `6.50 MB`):** Standard Git is optimal and requires no Git LFS configuration, allowing any researcher or evaluator to clone and run the simulator and test suite immediately without LFS bandwidth quotas.
- **Future Phase 3.3 Individual Household Artifacts:** The local `data/processed/mathura_individual_meters.parquet` (`114.95 MB`) and `data/processed/bareilly_individual_meters.parquet` (`203.98 MB`) exceed GitHub's `100 MB` per-file limit and are properly excluded via `.gitignore`. When the 18 individual checkpoints (`ind_*.pt`) and `individual_predictions.parquet` are exported from Colab in Phase 3.3, they should be attached as a versioned **GitHub Release Archive Asset** (`ind_transfer_package.zip`, which supports files up to `2 GB` without consuming Git LFS quota).

---

## 6. Final Staged-File Inventory (`195` Files on `main`)

All 195 staged files fall cleanly into the following directory groups:
- **Root Metadata & Configuration (`9` files):** `.env.example`, `.gitignore`, `.python-version`, `CITATION.cff`, `CONTRIBUTING.md`, `LICENSE`, `README.md`, `requirements.txt`, `config/config.yaml`
- **Audit Reports (`7` files in `AUDIT/`):** `APPLICATION_FUNCTIONALITY_REPORT.md`, `CLEANUP_PLAN.md`, `FINAL_RELEASE_VALIDATION.md`, `INFERENCE_VERIFICATION_REPORT.md`, `REPOSITORY_INVENTORY.md`, `RESEARCH_COMPLETENESS_REPORT.md`, `SECURITY_AUDIT_REPORT.md`
- **Interactive Applications (`11` files in `app/`):** `app/.gitkeep`, `app/app.py`, `app/build_dashboard_data.py`, `app/build_simulator_bundle.py`, `app/serve_dashboard.py`, `app/dashboard/index.html`, `app/dashboard/styles.css`, `app/dashboard/app.js`, `app/dashboard/simulator.js`, `app/dashboard/data.js`, `app/dashboard/simulator_bundle.js`
- **Verified Model Checkpoints (`19` files in `artifacts/checkpoints/`):** `.gitkeep` + `18` `agg_{mathura,bareilly}_{1D_CNN,LSTM,CNN_LSTM}_{consumption_only,consumption_weather_retro,consumption_weather_operational}.pt`
- **Empirical Metrics, Logs & Figures (`37` files in `artifacts/metrics/`):** `.gitkeep`, `training_results.json`, `baseline_benchmarks.json`, `aggregate_test_predictions.parquet`, `aggregate_test_predictions.csv`, `dm_test_results.csv`, `dm_test_summary.json`, `dm_significance_comparison.png`, `master_experiment_comparison.png`, `checkpoint_integrity_manifest.csv`, `compute_pilot_results.json`, `experiment_summary.csv`, `hac_lag_sensitivity.csv`, `phase_3_4_consolidated_results.csv`, `pre_training_audit_summary.json`, `5` EDA figures (`eda/fig1..fig5_*.png`), and `18` per-experiment training logs (`training_logs/agg_*.json`)
- **Dataset Directory (`6` files in `data/`):** `data/README.md`, `data/raw/.gitkeep`, `data/interim/.gitkeep`, `data/processed/.gitkeep`, `data/processed/mathura_features.parquet`, `data/processed/bareilly_features.parquet`
- **Documentation & Phase Audit Records (`55` files in `docs/`):** `.gitkeep`, `METHODOLOGY.md`, `DATASETS.md`, `MODEL_ARCHITECTURES.md`, `INFERENCE_GUIDE.md`, `REPRODUCIBILITY.md`, `LIMITATIONS.md`, `REFERENCES.md`, `KNOWN_ISSUES.md`, chronological phase reports (`PHASE_1_1` through `PHASE_3_4`), `docs/audit_results/*` (`9` files), and `docs/dataset_inventory/*` (`15` files)
- **Notebooks (`3` files in `notebooks/`):** `.gitkeep`, `01_exploratory_data_analysis.ipynb`, `02_colab_training_pipeline.ipynb`
- **Logs Directory (`1` file in `logs/`):** `logs/.gitkeep`
- **Pipeline & Evaluation Scripts (`11` files in `scripts/`):** `generate_aggregate_predictions.py`, `generate_individual_predictions.py`, `generate_phase_3_4_consolidated_results.py`, `package_colab_checkpoints.py`, `run_compute_pilot.py`, `run_diebold_mariano_evaluation.py`, `run_individual_panel_evaluation.py`, `test_hac_sensitivity.py`, `verify_imported_checkpoints.py`, `verify_metrics_reconciliation.py`
- **Core Source Package (`20` files in `src/`):** `__init__.py`, `acquire_weather.py`, `audit_boundary_purge.py`, `audit_raw_data.py`, `baselines.py`, `construct_targets.py`, `controlled_downloader.py`, `dataset.py`, `evaluate.py`, `generate_acquisition_report.py`, `generate_eda.py`, `generate_pre_training_audit_artifacts.py`, `panel_statistical_tests.py`, `reconcile_and_audit.py`, `simulator.py`, `statistical_tests.py`, `train.py`, `train_runner.py`, and `src/models/{__init__.py,cnn1d.py,lstm.py,cnn_lstm.py}`
- **Unit & Integration Tests (`8` files in `tests/`):** `.gitkeep`, `test_baselines_and_metrics.py`, `test_data_integrity.py`, `test_dataset_slicing.py`, `test_panel_statistical_tests.py`, `test_simulator.py`, `test_statistical_tests.py`, `test_target_construction_phase1_5.py`

### Explicitly Excluded Local Files (`!!` in `git status --ignored`, Preserved on Disk)
- `.venv/` and `__pycache__/` directories
- `data/raw/ceew/`, `data/raw/iiit_delhi/`, `data/raw/weather/`
- `data/processed/mathura_individual_meters.parquet`, `data/processed/bareilly_individual_meters.parquet`, `data/processed/delhi_*.parquet`, `data/processed/*_aggregate.parquet`
- `artifacts/checkpoints/ind_mathura_1D_CNN_consumption_only.pt` (unverified local pilot checkpoint)
- All 15 duplicate/one-off/presentation files listed in `AUDIT/CLEANUP_PLAN.md`

---

## 7. Remaining Technical & Operational Risks

1. **Pending Phase 3.3 Export of Individual Household Binary Checkpoints (`ind_*.pt`):**
   - While all 18 individual household models were trained and evaluated on Google Colab GPU (with full metrics recorded in `artifacts/metrics/training_results.json`), the 18 binary `ind_*.pt` files and `individual_predictions.parquet` remain pending export from Colab. This is transparently disclosed across `README.md`, `docs/LIMITATIONS.md`, and the simulator UI, which restricts live inference to the 18 verified aggregate checkpoints.
2. **PyTorch Deserialization Trust Boundary:**
   - PyTorch checkpoints use pickle serialization internally (`weights_only=False` is required to read embedded metadata dictionaries in `agg_*.pt`). This risk is mitigated by verifying SHA-256 digests (`artifacts/metrics/checkpoint_integrity_manifest.csv`), enforcing strict allowlist membership (`self._availability_by_id`), and verifying path containment (`ckpt_path.resolve().is_relative_to(self.checkpoints_dir.resolve())`) prior to `torch.load`. Users should never load untrusted third-party `.pt` files into `artifacts/checkpoints/`.
3. **Localhost Binding for `serve_dashboard.py`:**
   - `app/serve_dashboard.py` is designed as a single-user local research server bound to `127.0.0.1` with a 1 MB request limit and path-traversal protection; it should not be exposed directly to the public internet without a production reverse proxy and rate limiter.
