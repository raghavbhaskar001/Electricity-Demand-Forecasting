# Dataset Directory Guide (`data/`)

This directory stores raw smart-meter and meteorological files, intermediate aligned tables, and finalized 15-minute feature Parquet files used for model training, evaluation, and interactive simulation.

---

## 1. Directory Layout

```text
data/
├── raw/
│   ├── smart_meter/       # Raw CEEW (Mathura/Bareilly) and IIIT Delhi CSV archives
│   └── weather/           # Raw Open-Meteo ERA5-Land hourly CSV downloads
├── interim/               # Intermediate resampled and aligned tables
├── processed/
│   ├── mathura_features.parquet            # Tracked in Git (1.42 MB) — Mathura aggregate feeder series
│   ├── bareilly_features.parquet           # Tracked in Git (1.51 MB) — Bareilly aggregate feeder series
│   ├── delhi_features.parquet              # Local-only (0.04 MB) — Exploratory 3.85-day series
│   ├── mathura_individual_meters.parquet   # Ignored in Git (114.95 MB) — 38 household meters
│   └── bareilly_individual_meters.parquet  # Ignored in Git (203.98 MB) — 46 household meters
└── README.md
```

---

## 2. What Is Included in Version Control vs Ignored

To keep the repository lightweight while allowing the interactive prediction simulator and aggregate feeder evaluation scripts to run immediately after cloning:

- **Included in Git (`2.93 MB` total):**
  - `data/processed/mathura_features.parquet` (`35,127` rows at 15-minute intervals, May 2019 – May 2020)
  - `data/processed/bareilly_features.parquet` (`49,336` rows at 15-minute intervals, May 2019 – October 2020)
- **Excluded from Git (`.gitignore`):**
  - `data/raw/**` (`~1.05 GB` raw smart-meter CSV archives)
  - `data/processed/mathura_individual_meters.parquet` (`114.95 MB`, `1,568,487` meter-interval rows across 38 meters)
  - `data/processed/bareilly_individual_meters.parquet` (`203.98 MB`, `2,713,573` meter-interval rows across 46 meters)
  - `data/processed/delhi_features.parquet` (`39.6 KB`, exploratory only)

---

## 3. Upstream Data Sources & Licensing

| Cohort | Source | Geographic Coordinates | Native Resolution | Target Resolution | Upstream License / DOI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mathura** (Uttar Pradesh) | CEEW Smart Meter Dataset (*High frequency smart meter data from two districts in India*, V2.0) | `27.5220° N, 77.7044° E` | 3 minutes | 15 minutes (exact 5-reading mean) | CC0 1.0 Universal (`doi:10.7910/DVN/GOCHJH`) |
| **Bareilly** (Uttar Pradesh) | CEEW Smart Meter Dataset (*High frequency smart meter data from two districts in India*, V2.0) | `28.3656° N, 79.4779° E` | 3 minutes | 15 minutes (exact 5-reading mean) | CC0 1.0 Universal (`doi:10.7910/DVN/GOCHJH`) |
| **New Delhi** (Exploratory) | IIIT Delhi & UCLA Faculty Housing Dataset (*Information Strategies for Energy Conservation*, V1.0) | `28.5764° N, 77.2893° E` | 15 minutes (`Table_A2_Final.tab`) | 15 minutes | CC0 1.0 Universal (`doi:10.7910/DVN/7MEXN4`) |
| **ERA5-Land Weather** | Copernicus Climate Change Service (C3S) via Open-Meteo Historical API | District centroids above | 60 minutes | 15 minutes (backward as-of / 120h operational lag) | CC BY 4.0 / C3S (`doi:10.24381/cds.e2161bac`) |

---

## 4. Rebuilding the Full Processed Datasets from Raw Sources

1. **Acquire and verify raw smart-meter files** in `data/raw/`:
   - Run `python src/controlled_downloader.py` (or download directly from Harvard Dataverse `doi:10.7910/DVN/GOCHJH` v2.0 into `data/raw/ceew/` and `doi:10.7910/DVN/7MEXN4` v1.0 into `data/raw/iiit_delhi/`).
2. **Download ERA5-Land hourly weather data:**
   ```bash
   python src/acquire_weather.py
   ```
3. **Run raw data audit and construct 15-minute target and feature tables:**
   ```bash
   python src/audit_raw_data.py
   python src/construct_targets.py
   ```
   This step enforces strict `5 x 3-min` window completeness (`valid_obs_count == 5`), selects the fixed cohort of meters with `>= 90%` temporal coverage (`38` meters in Mathura, `46` meters in Bareilly), aligns retrospective (`t_weather <= T`) and operational (`t_weather <= T - 120h`) ERA5-Land weather covariates, and writes all Parquet files in `data/processed/`.
