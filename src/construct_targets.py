"""
Phase 1.5: Target Construction and Modelling Dataset Preparation Pipeline
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models

Key Responsibilities:
1. Target Construction:
   - IIIT Delhi: Recover physical kWh = exp(lnenergy); compute average kW = 4 * kWh for 15-min interval.
   - CEEW Mathura & Bareilly: Strict timestamp-aligned 5-subinterval mapping S(T) = {T-12, T-9, T-6, T-3, T}.
     Require all 5 observations present (nunique == 5). Compute kW = 4 * sum(t_kWh).
2. Operating State Preservation:
   - Preserve genuine zero-load (V >= 150V, I == 0A, kWh == 0 -> kW = 0.0) and outage flags.
   - Do NOT exclude zero consumption records.
3. Master Calendar Grid Alignment:
   - Align every meter to a continuous master 15-minute temporal grid.
   - Missing/incomplete windows are set to NaN and flagged with quality masks.
   - Contiguous sequence blocks identified so lookback sequences do not span across gaps.
4. Causal & Operational Weather Integration:
   - Retrospective Reanalysis Feature Set: Causal backward as-of merge (t_weather <= T).
   - Operational Availability Feature Set: 5-day publication lagged merge (t_weather <= T - 120h).
   - Track weather observation age and staleness in hours.
5. Cohort Aggregation:
   - IIIT Delhi: Primary ground truth strictly requires all 19 apartments (19/19).
     Sensitivity cohort evaluated at 18/19.
   - CEEW: Fixed cohort defined by sustained telemetry availability.
     Primary aggregate: >= 90% of cohort reporting complete window.
     Sensitivity aggregate: 85% to <90%.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TargetConstruction")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"
WEATHER_DIR = RAW_DIR / "weather"


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. IIIT DELHI PIPELINE
# ==============================================================================
def process_iiit_delhi():
    logger.info("=" * 80)
    logger.info("PROCESSING DATASET A: IIIT DELHI FACULTY HOUSING")
    logger.info("=" * 80)
    
    file_path = RAW_DIR / "iiit_delhi" / "Table_A2_Final.tab"
    df = pd.read_csv(file_path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["apt_id"] = "Apt_" + df["apt"].astype(int).astype(str)

    # 1. Recover physical energy (kWh) and average power demand (kW)
    # Energy in 15-min interval: kWh = exp(lnenergy)
    # Power demand: Average kW = 4 * kWh
    df["energy_15min_kwh"] = np.exp(df["lnenergy"])
    df["demand_kw"] = 4.0 * df["energy_15min_kwh"]
    
    # Quality filter per original methodology: pctmissing <= 10.0%
    df["is_valid_reading"] = (df["pctmissing"] <= 10.0) & np.isfinite(df["demand_kw"]) & (df["demand_kw"] >= 0.0)

    # Master calendar grid (15-minute resolution)
    t_start = df["timestamp"].min()
    t_end = df["timestamp"].max()
    master_grid = pd.date_range(t_start, t_end, freq="15min")
    logger.info(f"Master temporal grid: {len(master_grid):,} intervals ({t_start} to {t_end})")

    # Pivot to individual meter matrix
    valid_df = df[df["is_valid_reading"]]
    pivot_kw = valid_df.pivot(index="timestamp", columns="apt_id", values="demand_kw").reindex(master_grid)
    pivot_kwh = valid_df.pivot(index="timestamp", columns="apt_id", values="energy_15min_kwh").reindex(master_grid)
    pivot_valid = valid_df.pivot(index="timestamp", columns="apt_id", values="is_valid_reading").reindex(master_grid).fillna(False)

    # Aggregate Cohort Calculation
    # How many of the 19 apartments are valid at each timestamp?
    active_count = pivot_valid.sum(axis=1)
    
    # Primary ground truth: ALL 19 apartments valid
    primary_agg_kw = pivot_kw.sum(axis=1).where(active_count == 19, np.nan)
    primary_agg_kwh = pivot_kwh.sum(axis=1).where(active_count == 19, np.nan)

    # Sensitivity aggregate: exactly 18 apartments valid (observed partial sum)
    sensitivity_agg_kw = pivot_kw.sum(axis=1).where(active_count == 18, np.nan)

    aggregate_df = pd.DataFrame({
        "timestamp": master_grid,
        "active_apartment_count": active_count.values,
        "is_primary_cohort_valid": (active_count.values == 19),
        "is_sensitivity_cohort_valid": (active_count.values == 18),
        "aggregate_demand_kw_primary": primary_agg_kw.values,
        "aggregate_energy_kwh_primary": primary_agg_kwh.values,
        "aggregate_demand_kw_sensitivity": sensitivity_agg_kw.values,
    }).set_index("timestamp")

    logger.info(f"IIIT Delhi Aggregate Results:")
    logger.info(f"  Total grid timestamps: {len(master_grid):,}")
    logger.info(f"  Primary ground-truth valid intervals (19/19): {(active_count == 19).sum():,} ({(active_count == 19).mean()*100:.2f}%)")
    logger.info(f"  Sensitivity partial-sum valid intervals (18/19): {(active_count == 18).sum():,} ({(active_count == 18).mean()*100:.2f}%)")

    # Save individual meter table
    indiv_path = PROCESSED_DIR / "delhi_individual_meters.parquet"
    pivot_kw.reset_index(names=["timestamp"]).to_parquet(indiv_path, index=False)
    logger.info(f"Saved individual meter matrix to {indiv_path}")

    # Save aggregate table
    agg_path = PROCESSED_DIR / "delhi_aggregate.parquet"
    aggregate_df.reset_index().to_parquet(agg_path, index=False)
    logger.info(f"Saved aggregate demand matrix to {agg_path}")

    # Merge Weather Features
    weather_csv = WEATHER_DIR / "delhi_weather_hourly_2013_2014.csv"
    feature_df = merge_weather_features(aggregate_df.reset_index(), weather_csv, "delhi")
    
    feat_path = PROCESSED_DIR / "delhi_features.parquet"
    feature_df.to_parquet(feat_path, index=False)
    logger.info(f"Saved weather-aligned feature table to {feat_path}")

    return {
        "dataset": "IIIT Delhi",
        "total_intervals": len(master_grid),
        "primary_valid_intervals": int((active_count == 19).sum()),
        "sensitivity_valid_intervals": int((active_count == 18).sum()),
        "individual_meters": 19,
    }


# ==============================================================================
# 2. CEEW PIPELINE (MATHURA & BAREILLY)
# ==============================================================================
def process_ceew_district(district_name, file_specs):
    logger.info("=" * 80)
    logger.info(f"PROCESSING DATASET B: CEEW {district_name.upper()}")
    logger.info("=" * 80)

    district_dir = RAW_DIR / "ceew" / district_name
    
    # 1. Ingest and standardize telemetry across years
    dfs = []
    for spec in file_specs:
        fpath = district_dir / spec["filename"]
        logger.info(f"Loading {spec['filename']} ...")
        df_year = pd.read_csv(fpath, sep=spec["sep"], low_memory=False)
        
        # Standardize column mapping
        col_map = {}
        for c in df_year.columns:
            cl = c.lower()
            if "time" in cl:
                col_map[c] = "timestamp"
            elif "kwh" in cl:
                col_map[c] = "t_kwh"
            elif "volt" in cl:
                col_map[c] = "voltage_v"
            elif "curr" in cl or "amp" in cl:
                col_map[c] = "current_a"
            elif "freq" in cl or "hz" in cl:
                col_map[c] = "frequency_hz"
            elif "meter" in cl:
                col_map[c] = "meter_id"
        df_year = df_year.rename(columns=col_map)
        dfs.append(df_year[["timestamp", "meter_id", "t_kwh", "voltage_v", "current_a", "frequency_hz"]])

    raw_df = pd.concat(dfs, ignore_index=True)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df = raw_df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)
    logger.info(f"Total raw telemetry rows ingested for {district_name}: {len(raw_df):,}")

    # 2. Flag Operating States at 3-minute resolution
    raw_df["is_outage"] = (raw_df["voltage_v"] < 50.0) & (raw_df["current_a"] == 0.0) & (raw_df["t_kwh"] == 0.0)
    raw_df["is_zero_load"] = (raw_df["voltage_v"] >= 150.0) & (raw_df["current_a"] == 0.0) & (raw_df["t_kwh"] == 0.0)

    # 3. Fixed Cohort Identification (telemetry availability, not consumption)
    meter_records = raw_df.groupby("meter_id")["timestamp"].agg(["min", "max", "count"])
    # A meter belongs to the fixed longitudinal cohort if it has multi-year participation
    # (spans from 2019 into 2020/2021 with at least 50,000 telemetry packets)
    cohort_mask = (meter_records["count"] >= 50000) & (meter_records["min"].dt.year == 2019) & (meter_records["max"].dt.year >= 2020)
    fixed_cohort = sorted(meter_records[cohort_mask].index.tolist())
    logger.info(f"Fixed Longitudinal Cohort for {district_name.upper()} ({len(fixed_cohort)} meters): {fixed_cohort}")

    # 4. Strict 15-Minute Target Construction
    # Map 3-minute timestamps to closing 15-minute interval: T = ceil('15min')
    raw_df["window_close_t"] = raw_df["timestamp"].dt.ceil("15min")

    # Aggregate by (meter_id, window_close_t)
    logger.info("Performing strict 15-minute interval aggregation ...")
    agg_window = raw_df.groupby(["meter_id", "window_close_t"]).agg(
        subinterval_count=("timestamp", "nunique"),
        first_subinterval=("timestamp", "min"),
        last_subinterval=("timestamp", "max"),
        energy_15min_kwh=("t_kwh", "sum"),
        avg_voltage_v=("voltage_v", "mean"),
        avg_current_a=("current_a", "mean"),
        outage_count=("is_outage", "sum"),
        zero_load_count=("is_zero_load", "sum"),
    ).reset_index()

    # Completeness condition: exactly 5 subintervals, starting at T-12min, ending at T
    agg_window["expected_start"] = agg_window["window_close_t"] - pd.Timedelta(minutes=12)
    agg_window["is_complete_window"] = (
        (agg_window["subinterval_count"] == 5) &
        (agg_window["first_subinterval"] == agg_window["expected_start"]) &
        (agg_window["last_subinterval"] == agg_window["window_close_t"])
    )

    # Power conversion: Average kW = 4 * kWh
    agg_window["demand_kw"] = np.where(agg_window["is_complete_window"], 4.0 * agg_window["energy_15min_kwh"], np.nan)
    agg_window["is_window_outage"] = (agg_window["outage_count"] == 5)
    agg_window["is_window_zero_load"] = (agg_window["zero_load_count"] == 5)

    n_complete = agg_window["is_complete_window"].sum()
    n_total_windows = len(agg_window)
    logger.info(f"15-Minute Window Validation ({district_name.upper()}):")
    logger.info(f"  Total meter-windows formed: {n_total_windows:,}")
    logger.info(f"  Complete valid windows (all 5 timestamps present): {n_complete:,} ({n_complete/n_total_windows*100:.2f}%)")
    logger.info(f"  Rejected incomplete windows (blackout/missing packets): {n_total_windows - n_complete:,} ({(n_total_windows - n_complete)/n_total_windows*100:.2f}%)")

    # 5. Master Calendar Grid & Pivot Matrix
    t_start = agg_window["window_close_t"].min()
    t_end = agg_window["window_close_t"].max()
    master_grid = pd.date_range(t_start, t_end, freq="15min")
    logger.info(f"Master calendar grid for {district_name}: {len(master_grid):,} intervals ({t_start} to {t_end})")

    # Pivot individual meter matrices
    valid_windows = agg_window[agg_window["is_complete_window"]]
    pivot_kw = valid_windows.pivot(index="window_close_t", columns="meter_id", values="demand_kw").reindex(master_grid)
    pivot_valid = valid_windows.pivot(index="window_close_t", columns="meter_id", values="is_complete_window").reindex(master_grid).fillna(False)

    # Restrict to Fixed Cohort
    cohort_pivot_kw = pivot_kw[[m for m in fixed_cohort if m in pivot_kw.columns]]
    cohort_pivot_valid = pivot_valid[[m for m in fixed_cohort if m in pivot_valid.columns]]

    # 6. Cohort Aggregate Calculation
    n_cohort = len(fixed_cohort)
    active_in_cohort = cohort_pivot_valid.sum(axis=1)
    cohort_coverage_ratio = active_in_cohort / n_cohort

    # Primary analysis: >= 90% of fixed cohort active
    is_primary_eligible = cohort_coverage_ratio >= 0.90
    primary_agg_kw = cohort_pivot_kw.sum(axis=1).where(is_primary_eligible, np.nan)

    # Sensitivity analysis: 85% to <90% active
    is_sensitivity_eligible = (cohort_coverage_ratio >= 0.85) & (cohort_coverage_ratio < 0.90)
    sensitivity_agg_kw = cohort_pivot_kw.sum(axis=1).where(is_sensitivity_eligible, np.nan)

    aggregate_df = pd.DataFrame({
        "timestamp": master_grid,
        "fixed_cohort_size": n_cohort,
        "active_reporting_meters": active_in_cohort.values,
        "cohort_coverage_ratio": cohort_coverage_ratio.values,
        "is_primary_cohort_valid": is_primary_eligible.values,
        "is_sensitivity_cohort_valid": is_sensitivity_eligible.values,
        "aggregate_demand_kw_primary": primary_agg_kw.values,
        "aggregate_demand_kw_sensitivity": sensitivity_agg_kw.values,
    }).set_index("timestamp")

    logger.info(f"Cohort Aggregate Summary ({district_name.upper()}):")
    logger.info(f"  Fixed cohort size: {n_cohort} meters")
    logger.info(f"  Primary aggregate valid intervals (>=90% active): {is_primary_eligible.sum():,} ({is_primary_eligible.mean()*100:.2f}%)")
    logger.info(f"  Sensitivity aggregate valid intervals (85-89% active): {is_sensitivity_eligible.sum():,} ({is_sensitivity_eligible.mean()*100:.2f}%)")

    # Save individual meter table
    indiv_path = PROCESSED_DIR / f"{district_name}_individual_meters.parquet"
    pivot_kw.reset_index(names=["timestamp"]).to_parquet(indiv_path, index=False)
    logger.info(f"Saved individual meter matrix to {indiv_path}")

    # Save aggregate table
    agg_path = PROCESSED_DIR / f"{district_name}_aggregate.parquet"
    aggregate_df.reset_index().to_parquet(agg_path, index=False)
    logger.info(f"Saved aggregate demand matrix to {agg_path}")

    # Merge Weather Features
    weather_csv = WEATHER_DIR / f"{district_name}_weather_hourly_2019_2021.csv"
    feature_df = merge_weather_features(aggregate_df.reset_index(), weather_csv, district_name)
    
    feat_path = PROCESSED_DIR / f"{district_name}_features.parquet"
    feature_df.to_parquet(feat_path, index=False)
    logger.info(f"Saved weather-aligned feature table to {feat_path}")

    return {
        "dataset": f"CEEW {district_name.title()}",
        "total_intervals": len(master_grid),
        "fixed_cohort_size": n_cohort,
        "primary_valid_intervals": int(is_primary_eligible.sum()),
        "sensitivity_valid_intervals": int(is_sensitivity_eligible.sum()),
        "individual_meters": len(pivot_kw.columns),
    }


# ==============================================================================
# 3. CAUSAL & OPERATIONAL WEATHER MERGE
# ==============================================================================
def merge_weather_features(demand_df, weather_csv_path, loc_name):
    logger.info(f"Merging causal weather features for {loc_name} ...")
    weather_df = pd.read_csv(weather_csv_path)
    weather_df["weather_timestamp"] = pd.to_datetime(weather_df["time"])
    weather_df = weather_df.sort_values("weather_timestamp").reset_index(drop=True)

    var_cols = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "surface_pressure", "wind_speed_10m", "direct_radiation",
        "diffuse_radiation", "cloud_cover", "precipitation"
    ]

    demand_df = demand_df.sort_values("timestamp").reset_index(drop=True)

    # 1. Retrospective Reanalysis Features (Causal as-of merge: t_weather <= T)
    retro_weather = weather_df[["weather_timestamp"] + var_cols].copy()
    retro_merged = pd.merge_asof(
        demand_df[["timestamp"]],
        retro_weather,
        left_on="timestamp",
        right_on="weather_timestamp",
        direction="backward"
    )
    # Calculate age/staleness of retrospective observation
    retro_merged["weather_retrospective_age_minutes"] = (
        (retro_merged["timestamp"] - retro_merged["weather_timestamp"]).dt.total_seconds() / 60.0
    )
    retro_feature_cols = {col: f"retro_{col}" for col in var_cols}
    retro_merged = retro_merged.rename(columns=retro_feature_cols)

    # 2. Operational Availability Features (5-day publication latency: t_weather <= T - 120h)
    demand_df["operational_cutoff"] = demand_df["timestamp"] - pd.Timedelta(hours=120)
    op_merged = pd.merge_asof(
        demand_df[["timestamp", "operational_cutoff"]],
        weather_df[["weather_timestamp"] + var_cols],
        left_on="operational_cutoff",
        right_on="weather_timestamp",
        direction="backward"
    )
    op_merged["weather_operational_age_hours"] = (
        (op_merged["timestamp"] - op_merged["weather_timestamp"]).dt.total_seconds() / 3600.0
    )
    op_feature_cols = {col: f"operational_{col}" for col in var_cols}
    op_merged = op_merged.rename(columns=op_feature_cols)

    # Combine into unified feature table
    combined = demand_df.copy()
    for col in retro_feature_cols.values():
        combined[col] = retro_merged[col]
    combined["weather_retrospective_age_minutes"] = retro_merged["weather_retrospective_age_minutes"]

    for col in op_feature_cols.values():
        combined[col] = op_merged[col]
    combined["weather_operational_age_hours"] = op_merged["weather_operational_age_hours"]

    # Add Calendar Features
    combined["hour_of_day"] = combined["timestamp"].dt.hour
    combined["minute_of_hour"] = combined["timestamp"].dt.minute
    combined["day_of_week"] = combined["timestamp"].dt.dayofweek
    combined["is_weekend"] = combined["day_of_week"].isin([5, 6]).astype(int)
    combined["month"] = combined["timestamp"].dt.month

    # Cyclical encodings
    combined["hour_sin"] = np.sin(2 * np.pi * combined["hour_of_day"] / 24.0)
    combined["hour_cos"] = np.cos(2 * np.pi * combined["hour_of_day"] / 24.0)
    combined["day_sin"] = np.sin(2 * np.pi * combined["day_of_week"] / 7.0)
    combined["day_cos"] = np.cos(2 * np.pi * combined["day_of_week"] / 7.0)
    combined["month_sin"] = np.sin(2 * np.pi * combined["month"] / 12.0)
    combined["month_cos"] = np.cos(2 * np.pi * combined["month"] / 12.0)

    # Drop intermediate column
    if "operational_cutoff" in combined.columns:
        combined = combined.drop(columns=["operational_cutoff"])

    logger.info(f"Constructed {len(combined.columns)} total features for {len(combined):,} intervals.")
    return combined


# ==============================================================================
# 4. CONTIGUOUS SEQUENCE BLOCK IDENTIFICATION
# ==============================================================================
def compute_contiguous_sequence_blocks(series, lookback_len=16, horizon_len=1):
    """
    Computes valid (X, y) sequence indices that do not cross invalid/missing intervals.
    series: pd.Series of values with NaN where invalid.
    """
    valid_mask = series.notna().values
    total_len = len(series)
    window_size = lookback_len + horizon_len
    valid_indices = []

    for i in range(total_len - window_size + 1):
        window_valid = valid_mask[i : i + window_size]
        if window_valid.all():
            valid_indices.append(i)

    return len(valid_indices)


def run_all_target_construction():
    ensure_dirs()
    summary_results = []

    # 1. IIIT Delhi
    delhi_res = process_iiit_delhi()
    summary_results.append(delhi_res)

    # 2. CEEW Mathura
    mathura_specs = [
        {"year": 2019, "filename": "CEEW - Smart meter data Mathura 2019.csv", "sep": ","},
        {"year": 2020, "filename": "CEEW - Smart meter data Mathura 2020.csv", "sep": ","},
        {"year": 2021, "filename": "CEEW - Smart meter data Mathura 2021.tab", "sep": "\t"},
    ]
    mathura_res = process_ceew_district("mathura", mathura_specs)
    summary_results.append(mathura_res)

    # 3. CEEW Bareilly
    bareilly_specs = [
        {"year": 2019, "filename": "CEEW - Smart meter data Bareilly 2019.tab", "sep": "\t"},
        {"year": 2020, "filename": "CEEW - Smart meter data Bareilly 2020.csv", "sep": ","},
        {"year": 2021, "filename": "CEEW - Smart meter data Bareilly 2021.csv", "sep": ","},
    ]
    bareilly_res = process_ceew_district("bareilly", bareilly_specs)
    summary_results.append(bareilly_res)

    # Summary table
    sum_df = pd.DataFrame(summary_results)
    sum_csv = DOCS_DIR / "target_construction_summary.csv"
    sum_df.to_csv(sum_csv, index=False)
    logger.info(f"\nTarget Construction Complete. Summary saved to {sum_csv}")
    print(sum_df.to_string(index=False))


if __name__ == "__main__":
    run_all_target_construction()
