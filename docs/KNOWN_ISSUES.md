# Project Known Issues & Backlog

**Project Title:** Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models  
**Last Updated:** 2026-11-09  
**Purpose:** Central tracking repository for non-critical observations, minor empirical discrepancies, and backlog items that do not block core workflow execution.

---

## 1. Non-Critical Dataset Discrepancies & Anomalies

| ID | Category | Item Description | Severity | Impact on Primary Workflow | Status / Mitigation |
|---|---|---|---|---|---|
| **ISS-001** | Dataverse Ingestion | Dataverse repository API reports MD5 for original uploaded Stata `.dta` file (`56920985...`) rather than served tabular `.tab` file (`faf36e8b...`) for IIIT Delhi `Table_A2_Final.tab`. | Low | None. Both original file MD5 and tabular S3 ETag/MD5 verified and recorded. | Resolved / Documented in manifest |
| **ISS-002** | Temporal Gaps | Annual boundary transitions in CEEW Mathura (2019→2020 & 2020→2021): At global dataset level, timestamps span continuously across midnight ($\Delta t = 3\text{ min}$). At individual meter level, however, only 10/38 meters transition seamlessly in 2019→2020 (median gap 168 h / 7 days), and 15/35 meters in 2020→2021 (median gap 24 h). | Medium | Requires strict meter-specific master grid alignment to avoid false interpolation across meter-level New Year gaps. | Rigorously characterized; mitigated by master temporal grid masking |
| **ISS-003** | Temporal Gaps | Annual boundary transitions in CEEW Bareilly (2019→2020 & 2020→2021): Global file span is continuous ($\Delta t = 3\text{ min}$). At meter level, 29/46 meters (63%) in 2019→2020 and 30/38 meters (79%) in 2020→2021 have seamless 3-min transitions, while remaining meters exhibit multi-day logging gaps. | Medium | Addressed by per-meter calendar masks. | Rigorously characterized; mitigated by master temporal grid masking |
| **ISS-004** | Dataset Truncation | CEEW 2021 telemetry records terminate in February/March 2021 rather than full year. | Low | 2019 and 2020 provide complete multi-season annual cycles; 2021 used as out-of-sample forward test set. | Documented in audit specification |
| **ISS-005** | Spatial Granularity | Open-Meteo ERA5-Land centroid offsets from exact study sites: Delhi ~3.96 km, Mathura ~4.46 km, Bareilly ~4.65 km. | Low | Negligible for macro-atmospheric variables (temperature, humidity, solar radiation) at 0.1° grid resolution. | Documented in inventory register |
| **ISS-006** | Cohort Synchronicity | IIIT Delhi primary ground-truth cohort (all 19 apartments synchronously valid) spans 553 intervals (96 contiguous sequences of L=16). Staggered deployment dates explain lower synchronous count. | Low | Sensitivity cohort (18/19 apts, 229 sequences) and individual apartment models (262,680 sequences) provide full training depth. | Characterized in Phase 1.5 Report |
| **ISS-007** | Operational Weather Latency | ERA5-Land reanalysis has ~5-day publication latency; operational deployment simulation enforces 120-hour backward lag. | Low | Retrospective reanalysis and operational availability feature sets are maintained separately in feature tables. | Implemented in Phase 1.5 Pipeline |
| **ISS-008** | Statistical Testing Overlap | Contiguous sliding sequence windows ($L=16, H=1$) share 15 historical observations across adjacent steps, causing autocorrelation in error residuals. | Medium | Standard paired t-tests assuming i.i.d. observations are invalid. Hypothesis testing must use Diebold-Mariano tests with autocorrelation adjustment, block bootstrap, or non-overlapping evaluation subsampling. | Documented in Phase 2.0 QA Report |
| **ISS-009** | Individual Sequence Census | Initial Phase 2.0 report draft displayed tabular discrepancies in unpurged split component totals (1,673,894 vs 1,536,402 in Mathura; 2,794,030 vs 2,668,842 in Bareilly). | Low | Fully resolved. Verified that Phase 1.5 counts (1,536,402 and 2,668,842) are the exact ground truth. Generated reproducible meter-wise census (`docs/meter_wise_sequence_census.csv`) proving exact 0-discrepancy balance across Train, Purge 1, Val, Purge 2, and Test splits. | Reconciled in Phase 2.0 Final QA |

---

## 2. Infrastructure & Environment Backlog

| ID | Category | Item Description | Severity | Status / Workaround |
|---|---|---|---|---|
| **ENV-001** | PyTorch Wheels | Local Python version is 3.14.7; earlier prebuilt PyTorch wheels required Python $\le 3.12$ at project start, though `torch==2.15.0+cpu` is now installed and verified locally. | Low | Local environment (`Python 3.14.7`, `torch==2.15.0+cpu`) verified across all 18 aggregate checkpoints (`agg_*.pt`) and 44 unit tests; Google Colab GPU was used for full 36-model training runs. | Resolved |
| **ENV-002** | WMI/CIM Cmdlets | Standard PowerShell CIM cmdlets (`Get-CimInstance`) restricted under sandboxed non-elevated user token. | Low | Replaced with native .NET P/Invoke `GlobalMemoryStatusEx` and standard environment variables. | Resolved |

---

## 3. Backlog & Sensitivity Exploration Tasks

- [ ] **TSK-001:** Evaluate impact of $18/19$ vs $19/19$ apartment cohort completeness threshold on IIIT Delhi aggregate forecast error.
- [ ] **TSK-002:** Compare 85% vs 90% active meter cohort threshold on CEEW Mathura aggregate forecast stability.
- [ ] **TSK-003:** Assess lag sensitivity of weather covariates ($T-15\text{min}$, $T-1\text{h}$, $T-2\text{h}$) against instantaneous reanalysis interpolation.

