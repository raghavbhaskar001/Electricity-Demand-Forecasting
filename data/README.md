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
| **Mathura** (Uttar Pradesh) | Council on Energy, Environment and Water (CEEW) Smart Meter Dataset | `27.4924° N, 77.6737° E` | 3 minutes | 15 minutes (exact 5-reading mean) | CC0 1.0 Public Domain (`doi:10.7910/DVN/N3615P`) |
| **Bareilly** (Uttar Pradesh) | Council on Energy, Environment and Water (CEEW) Smart Meter Dataset | `28.3670° N, 79.4304° E` | 3 minutes | 15 minutes (exact 5-reading mean) | CC0 1.0 Public Domain (`doi:10.7910/DVN/N3615P`) |
| **New Delhi** (Exploratory) | IIIT Delhi Residential Electric Power Dataset | `28.5450° N, 77.2732° E` | 1 minute | 15 minutes | Academic research use (`doi:10.6084/m9.figshare.c.4490387`) |
| **ERA5-Land Weather** | Copernicus Climate Change Service (C3S) via Open-Meteo Historical API | District centroids above | 60 minutes | 15 minutes (cubic-spline / step interpolation) | C3S / CC BY 4.0 (`doi:10.24381/cds.e2161bac`) |

---

## 4. Rebuilding the Full Processed Datasets from Raw Sources

1. **Place raw smart-meter files** in `data/raw/smart_meter/`:
   - Download the CEEW smart-meter CSV archive from Harvard Dataverse (`doi:10.7910/DVN/N3615P`) into `data/raw/smart_meter/`.
   - (Optional) Place the IIIT Delhi CSV archive in `data/raw/smart_meter/delhi/` for the exploratory cohort check.
2. **Download ERA5-Land hourly weather data:**
   ```bash
   python scripts/download_weather.py
   ```
3. **Process and filter 3-minute smart-meter records into 15-minute intervals:**
   ```bash
   python scripts/process_smart_meter.py
   ```
   This step enforces strict `5 x 3-min` window completeness (`valid_obs_count == 5`) and selects the fixed cohort of meters with `>= 90%` temporal coverage (`38` meters in Mathura, `46` meters in Bareilly).
4. **Align load and meteorological covariates (retrospective and 120-hour operational lag):**
   ```bash
   python scripts/align_weather_and_load.py
   ```
   This generates all Parquet files in `data/processed/`.
