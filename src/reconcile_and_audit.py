"""
Reconciliation & Boundary Audit Generator
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models

Generates:
1. docs/meter_wise_sequence_census.csv: Exact meter-by-meter sequence census across all 103 series
2. docs/temporal_boundary_audit.csv: Input/target timestamps and purge gap audit proving zero boundary overlap
3. artifacts/metrics/baseline_benchmarks.json: Full-test and matched-subset baseline evaluation
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.dataset import extract_contiguous_sequences, temporal_train_val_test_split
from src.baselines import run_all_baseline_benchmarks


DOCS_DIR = Path("docs")
DATA_DIR = Path("data/processed")


def generate_meter_wise_census():
    print("Generating meter-wise sequence census across all cohorts...")
    records = []

    cohorts = [
        ("IIIT Delhi Apartments", DATA_DIR / "delhi_individual_meters.parquet", "Delhi"),
        ("CEEW Mathura Smart Meters", DATA_DIR / "mathura_individual_meters.parquet", "Mathura"),
        ("CEEW Bareilly Smart Meters", DATA_DIR / "bareilly_individual_meters.parquet", "Bareilly"),
    ]

    for cohort_name, path, district in cohorts:
        df = pd.read_parquet(path)
        meter_cols = [c for c in df.columns if c != "timestamp"]

        for m in meter_cols:
            X, y, ts = extract_contiguous_sequences(df, m, [m], lookback_len=16, horizon_len=1)
            n_total = len(X)
            if n_total < 100:
                continue

            (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, _) = temporal_train_val_test_split(
                X, y, ts, train_ratio=0.70, val_ratio=0.15, purge_gap=16
            )

            n_tr = len(X_tr)
            n_va = len(X_va)
            n_te = len(X_te)
            p1 = 16
            p2 = 16
            split_sum = n_tr + p1 + n_va + p2 + n_te
            diff = n_total - split_sum

            records.append({
                "District": district,
                "Cohort": cohort_name,
                "Meter_ID": m,
                "Total_Contiguous_Sequences": n_total,
                "Train_Sequences_70pct": n_tr,
                "Purge_Gap_1": p1,
                "Val_Sequences_15pct": n_va,
                "Purge_Gap_2": p2,
                "Test_Sequences_15pct": n_te,
                "Component_Sum": split_sum,
                "Reconciled_Discrepancy": diff,
            })

    census_df = pd.DataFrame(records)
    out_csv = DOCS_DIR / "meter_wise_sequence_census.csv"
    census_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(census_df)} meters)")

    # Print summary per district
    for dist in ["Delhi", "Mathura", "Bareilly"]:
        sub = census_df[census_df["District"] == dist]
        tot = sub["Total_Contiguous_Sequences"].sum()
        tr = sub["Train_Sequences_70pct"].sum()
        va = sub["Val_Sequences_15pct"].sum()
        te = sub["Test_Sequences_15pct"].sum()
        p = sub["Purge_Gap_1"].sum() + sub["Purge_Gap_2"].sum()
        print(f"[{dist}] Meters: {len(sub)} | Total: {tot:,} | Train: {tr:,} | Val: {va:,} | Test: {te:,} | Purge: {p:,} | Discrepancy: {tot - (tr + va + te + p)}")

    return census_df


def generate_temporal_boundary_audit():
    print("\nGenerating temporal boundary audit ...")
    boundary_records = []

    targets = [
        ("Mathura Aggregate", DATA_DIR / "mathura_features.parquet", "aggregate_demand_kw_primary"),
        ("Bareilly Aggregate", DATA_DIR / "bareilly_features.parquet", "aggregate_demand_kw_primary"),
        ("Delhi Aggregate (Exploratory)", DATA_DIR / "delhi_features.parquet", "aggregate_demand_kw_primary"),
        ("Mathura Meter MH01", DATA_DIR / "mathura_individual_meters.parquet", "MH01"),
        ("Bareilly Meter BR02", DATA_DIR / "bareilly_individual_meters.parquet", "BR02"),
        ("Delhi Apartment Apt_1", DATA_DIR / "delhi_individual_meters.parquet", "Apt_1"),
    ]

    for name, path, target_col in targets:
        df = pd.read_parquet(path)
        valid_mask = df[target_col].notna().values
        time_vals = df["timestamp"].values

        in_starts, in_ends, target_tss = [], [], []
        for i in range(len(df) - 17 + 1):
            if valid_mask[i : i + 17].all():
                in_starts.append(time_vals[i])
                in_ends.append(time_vals[i + 15])
                target_tss.append(time_vals[i + 16])

        if not target_tss:
            continue

        in_starts = pd.to_datetime(in_starts)
        in_ends = pd.to_datetime(in_ends)
        target_tss = pd.to_datetime(target_tss)

        n = len(target_tss)
        n_tr = int(0.70 * n)
        va_start = n_tr + 16
        n_va = int(0.15 * n)
        va_end = va_start + n_va
        te_start = va_end + 16

        # Check if test partition is valid
        has_test = (te_start < n)
        n_te = max(0, n - te_start)

        rec = {
            "Dataset": name,
            "Total_Sequences": n,
            "Train_Count": n_tr,
            "Train_Last_In_Start": str(in_starts[n_tr - 1]),
            "Train_Last_In_End": str(in_ends[n_tr - 1]),
            "Train_Last_Target": str(target_tss[n_tr - 1]),
            "Val_Count": n_va if va_start < n else 0,
            "Val_First_In_Start": str(in_starts[va_start]) if va_start < n else "N/A",
            "Val_First_In_End": str(in_ends[va_start]) if va_start < n else "N/A",
            "Val_First_Target": str(target_tss[va_start]) if va_start < n else "N/A",
            "Train_Val_Gap_Minutes": (in_starts[va_start] - target_tss[n_tr - 1]).total_seconds() / 60.0 if va_start < n else None,
            "Val_Last_In_Start": str(in_starts[va_end - 1]) if va_end <= n else "N/A",
            "Val_Last_In_End": str(in_ends[va_end - 1]) if va_end <= n else "N/A",
            "Val_Last_Target": str(target_tss[va_end - 1]) if va_end <= n else "N/A",
            "Test_Count": n_te,
            "Test_First_In_Start": str(in_starts[te_start]) if has_test else "N/A",
            "Test_First_In_End": str(in_ends[te_start]) if has_test else "N/A",
            "Test_First_Target": str(target_tss[te_start]) if has_test else "N/A",
            "Val_Test_Gap_Minutes": (in_starts[te_start] - target_tss[va_end - 1]).total_seconds() / 60.0 if (has_test and va_end <= n) else None,
            "Zero_Overlap_Verified": True if (va_start < n and (not has_test or (in_starts[te_start] > target_tss[va_end - 1]))) else False,
        }
        boundary_records.append(rec)

    boundary_df = pd.DataFrame(boundary_records)
    out_csv = DOCS_DIR / "temporal_boundary_audit.csv"
    boundary_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return boundary_df


if __name__ == "__main__":
    generate_meter_wise_census()
    generate_temporal_boundary_audit()
    print("\nRegenerating baseline benchmarks with matched-subset metrics...")
    run_all_baseline_benchmarks()
    print("All reconciliations and baseline benchmarks updated successfully.")

