# Security Audit and Privacy Hardening Report

**Project:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Author:** Raghav Bhaskar — Department of Electrical Engineering, Thapar Institute of Engineering & Technology, Patiala  
**Audit Date:** 2026-11-09  

---

## 1. Secrets & Credential Scan Results

A full regex and AST scan was executed across all Python (`.py`), JavaScript (`.js`), HTML (`.html`), CSS (`.css`), JSON (`.json`), YAML (`.yaml`), Markdown (`.md`), CSV (`.csv`), and Jupyter Notebook (`.ipynb`) files in the repository.

| Scan Category | Pattern / Scope | Matches Found | Status |
| :--- | :--- | :---: | :--- |
| **AWS Access Keys / Secrets** | `AKIA[0-9A-Z]{16}` | `0` | **PASS** |
| **GitHub Personal / OAuth Tokens** | `gh[pousr]_[A-Za-z0-9_]{20,}` | `0` | **PASS** |
| **Google Cloud / AI Studio Keys** | `AIza[0-9A-Za-z\-_]{35}` | `0` | **PASS** |
| **RSA / EC / OpenSSH Private Keys** | `-----BEGIN ... PRIVATE KEY-----` | `0` | **PASS** |
| **Hardcoded API Key / Password Assignments** | `(api_key\|secret_key\|access_token\|password)\s*[:=]\s*['"]...['"]` | `0` | **PASS** |
| **Tracked `.env` or Credential Files** | `.env`, `.env.*`, `credentials.json`, `secrets.yaml`, `*.pem`, `*.key` | `0` | **PASS** |
| **Git Commit History Scan** | `git log --all --full-history` | `0` commits exist yet | **PASS** (Clean slate before initial commit) |
| **Hardcoded Local Filesystem Paths** | `<repo-root>` and `...` | Found in 14 files (`notebooks/02_colab_training_pipeline.ipynb`, `scripts/package_colab_checkpoints.py`, and 12 `docs/PHASE_*.md` reports) | **REMEDIATED** (Converted to portable relative paths) |

---

## 2. Environment Isolation (`.env.example` & `.gitignore`)

- Created `.env.example` containing only safe placeholder variable names (`DASHBOARD_HOST=127.0.0.1`, `DASHBOARD_PORT=8505`, `PROJECT_RANDOM_SEED=42`, `DATAVERSE_BASE_URL=https://dataverse.harvard.edu`, `OPEN_METEO_ARCHIVE_URL=https://archive-api.open-meteo.com/v1/archive`, `DATAVERSE_API_TOKEN=your_optional_dataverse_token_here`).
- Updated `.gitignore` so `.env` and `.env.*` (with explicit `!/.env.example` negation) are strictly excluded from version control.
- No proprietary API keys are required to run the dashboard, simulator, or test suite.

---

## 3. Authentication & Cryptographic Usage

- The dashboard and prediction simulator run locally without a user authentication layer (none was added unnecessarily).
- `hashlib.sha256` is used strictly for dataset, artifact, and model checkpoint file-integrity verification (`src/controlled_downloader.py`, `app/build_dashboard_data.py`, `src/simulator.py`), never as a password-hashing mechanism or secret obfuscation layer.

---

## 4. Data Privacy & Licensing Audit

| Dataset / Resource | Provenance & DOI | License | Personal / Household Privacy Assessment | Public Repository Handling |
| :--- | :--- | :--- | :--- | :--- |
| **CEEW Smart Meter Dataset (Mathura & Bareilly)** | Harvard Dataverse `doi:10.7910/DVN/GOCHJH` (v2.0) | **CC0 1.0 Universal** (Public Domain Dedication) | Uses anonymized meter codes (`MH01..MH47`, `BR02..BR52`) with no names, street addresses, or household GPS coordinates. | Raw files (`1.02 GB`) and individual household time-series parquets (`*_individual_meters.parquet`) are **excluded** via `.gitignore`. Only aggregated feeder-level feature parquets (`mathura_features.parquet`, `bareilly_features.parquet`, `2.93 MB` total) are retained for reproducible testing and simulator execution. |
| **IIIT Delhi Faculty Housing Dataset** | Harvard Dataverse `doi:10.7910/DVN/7MEXN4` (v1.0) | **CC0 1.0 Universal** (Public Domain Dedication) | Anonymized apartment numbers (`apt 1..19`). | Raw and processed files **excluded** via `.gitignore`; acquisition documented in `data/README.md` and `docs/DATASETS.md`. |
| **Open-Meteo ERA5-Land Reanalysis** | Open-Meteo / Copernicus `doi:10.24381/cds.e2161bac` | **CC BY 4.0 International** | Gridded regional meteorological variables (177–224 m elevation grid cells); zero personal data. | Attributed to Open-Meteo.com and ECMWF Copernicus Climate Change Service in `README.md`, `CITATION.cff`, and `docs/DATASETS.md`. |

---

## 5. Application & Code Security Hardening

| Check Area | Inspection & Hardening Applied | Status |
| :--- | :--- | :--- |
| **Unsafe File Uploads / DoS** | `app/serve_dashboard.py` enforces `MAX_BODY_BYTES = 1_048_576` (`1 MB`) on `POST /api/*`, returning `HTTP 413 Payload Too Large` before reading oversized payloads into memory. CSV uploads are parsed strictly as numerical text via Python `csv.reader` and never written to disk or executed. | **HARDENED & VERIFIED** |
| **Path Traversal & Directory Listing** | `app/serve_dashboard.py` resolves all requested static paths via `Path.resolve().is_relative_to(DASH_DIR)` and disables directory listing (`list_directory` returns `403 Forbidden`). | **HARDENED & VERIFIED** |
| **Checkpoint Deserialization (`torch.load`)** | `src/simulator.py` verifies that `exp_id` belongs to the 18-model supported allowlist and validates `ckpt_path.resolve().is_relative_to(self.checkpoints_dir.resolve())` before loading local `.pt` files. User inputs never control checkpoint file paths. | **HARDENED & VERIFIED** |
| **Network Exposure & HTTP Headers** | Server binds strictly to loopback `127.0.0.1` by default (not `0.0.0.0`), sets `X-Content-Type-Options: nosniff` and `X-Frame-Options: SAMEORIGIN`, and suppresses internal stack traces on `500` responses. | **HARDENED & VERIFIED** |
| **Command / SQL Injection & Telemetry** | Zero `subprocess`/`os.system`/`eval` calls on user input; zero SQL databases; zero external network requests or telemetry in the dashboard (`index.html`, `app.js`, `simulator.js`). | **VERIFIED** |
