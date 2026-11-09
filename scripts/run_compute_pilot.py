"""
Compute Pilot Benchmark Runner for Short-Term Electricity Demand Forecasting
Records actual measured epoch duration, throughput, memory, and validation loss.
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.dataset import (
    FEATURE_SETS,
    extract_contiguous_sequences,
    temporal_train_val_test_split,
)
from src.evaluate import compute_forecasting_metrics
from src.train_runner import FastTensorDataset, get_model, train_epoch, evaluate_loss, PooledMeterTargetScaler


def run_compute_pilot():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*75}\nSTARTING COMPUTE PILOT BENCHMARK\nDevice: {device.upper()} | PyTorch: {torch.__version__}\n{'='*75}")
    sys.stdout.flush()

    pilot_metrics = {}

    # --------------------------------------------------------------------------
    # PILOT 1: Representative Aggregate Experiment (agg_mathura_1D_CNN_consumption_only)
    # --------------------------------------------------------------------------
    print("\n--- PILOT 1: Representative Aggregate Experiment ---")
    df_feat = pd.read_parquet("data/processed/mathura_features.parquet")
    f_cols = FEATURE_SETS["consumption_only"]
    X, y, ts = extract_contiguous_sequences(df_feat, "aggregate_demand_kw_primary", f_cols, lookback_len=16, horizon_len=1)
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, _) = temporal_train_val_test_split(
        X, y, ts, train_ratio=0.70, val_ratio=0.15, purge_gap=16
    )

    targ_scaler = StandardScaler()
    feat_scaler = StandardScaler()
    X_tr_s = feat_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va_s = feat_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    y_tr_s = targ_scaler.fit_transform(y_tr)
    y_va_s = targ_scaler.transform(y_va)

    tr_loader = DataLoader(FastTensorDataset(X_tr_s, y_tr_s), batch_size=64, shuffle=True)
    va_loader = DataLoader(FastTensorDataset(X_va_s, y_va_s), batch_size=128, shuffle=False)

    model_agg = get_model("1D_CNN", in_features=1).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model_agg.parameters(), lr=1e-3, weight_decay=1e-4)

    agg_epoch_times = []
    agg_val_losses = []
    agg_throughput = []

    print(f"Dataset Size: Train={len(X_tr_s)}, Val={len(X_va_s)} | Running 3 Pilot Epochs ...")
    for ep in range(1, 4):
        t0 = time.time()
        tr_loss = train_epoch(model_agg, tr_loader, optimizer, criterion, device)
        dur = time.time() - t0
        va_loss = evaluate_loss(model_agg, va_loader, criterion, device)
        thru = len(X_tr_s) / max(dur, 1e-4)

        agg_epoch_times.append(dur)
        agg_val_losses.append(va_loss)
        agg_throughput.append(thru)
        print(f"Epoch {ep:02d} | Duration: {dur:.3f}s | Throughput: {thru:.1f} samples/s | Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f}")
        sys.stdout.flush()

    pilot_metrics["aggregate_pilot"] = {
        "experiment_id": "agg_mathura_1D_CNN_consumption_only",
        "device": device,
        "train_samples": len(X_tr_s),
        "batch_size": 64,
        "mean_epoch_duration_sec": round(float(np.mean(agg_epoch_times)), 4),
        "mean_throughput_samples_per_sec": round(float(np.mean(agg_throughput)), 1),
        "final_val_loss": round(float(agg_val_losses[-1]), 6),
        "epoch_durations": [round(t, 4) for t in agg_epoch_times],
        "val_losses": [round(l, 6) for l in agg_val_losses]
    }

    # --------------------------------------------------------------------------
    # PILOT 2: Representative Pooled Individual Experiment (ind_mathura_1D_CNN_consumption_only)
    # --------------------------------------------------------------------------
    print("\n--- PILOT 2: Representative Pooled Individual Experiment ---")
    df_ind = pd.read_parquet("data/processed/mathura_individual_meters.parquet")
    meters = [c for c in df_ind.columns if c != "timestamp"][:5] # Representative 5-meter panel pilot
    print(f"Extracting pooled sequences for representative 5-meter panel: {meters} ...")

    pool_X_tr, pool_y_tr = [], []
    pool_X_va, pool_y_va = [], []
    meter_test_slices = []
    curr_te = 0

    for m in meters:
        sub_df = df_ind[["timestamp", m]].copy()
        sub_df["demand_target"] = sub_df[m]
        X_m, y_m, ts_m = extract_contiguous_sequences(sub_df, "demand_target", ["demand_target"], lookback_len=16, horizon_len=1)
        (X_tr_m, y_tr_m, _), (X_va_m, y_va_m, _), (X_te_m, y_te_m, _) = temporal_train_val_test_split(
            X_m, y_m, ts_m, train_ratio=0.70, val_ratio=0.15, purge_gap=16
        )
        s_m = StandardScaler()
        y_tr_m_s = s_m.fit_transform(y_tr_m)
        y_va_m_s = s_m.transform(y_va_m)
        X_tr_m_s = s_m.transform(X_tr_m.reshape(-1, 1)).reshape(X_tr_m.shape)
        X_va_m_s = s_m.transform(X_va_m.reshape(-1, 1)).reshape(X_va_m.shape)

        pool_X_tr.append(X_tr_m_s)
        pool_y_tr.append(y_tr_m_s)
        pool_X_va.append(X_va_m_s)
        pool_y_va.append(y_va_m_s)

    X_tr_p = np.vstack(pool_X_tr)
    y_tr_p = np.vstack(pool_y_tr)
    X_va_p = np.vstack(pool_X_va)
    y_va_p = np.vstack(pool_y_va)

    print(f"Pooled Pilot Size: Train={len(X_tr_p)}, Val={len(X_va_p)} | Batch Size=256 | Running 2 Pilot Epochs ...")
    tr_loader_p = DataLoader(FastTensorDataset(X_tr_p, y_tr_p), batch_size=256, shuffle=True)
    va_loader_p = DataLoader(FastTensorDataset(X_va_p, y_va_p), batch_size=512, shuffle=False)

    model_ind = get_model("1D_CNN", in_features=1).to(device)
    optimizer_ind = torch.optim.AdamW(model_ind.parameters(), lr=1e-3, weight_decay=1e-4)

    ind_epoch_times = []
    ind_val_losses = []
    ind_throughput = []

    for ep in range(1, 3):
        t0 = time.time()
        tr_loss = train_epoch(model_ind, tr_loader_p, optimizer_ind, criterion, device)
        dur = time.time() - t0
        va_loss = evaluate_loss(model_ind, va_loader_p, criterion, device)
        thru = len(X_tr_p) / max(dur, 1e-4)

        ind_epoch_times.append(dur)
        ind_val_losses.append(va_loss)
        ind_throughput.append(thru)
        print(f"Epoch {ep:02d} | Duration: {dur:.3f}s | Throughput: {thru:.1f} samples/s | Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f}")
        sys.stdout.flush()

    pilot_metrics["individual_pooled_pilot"] = {
        "experiment_id": "ind_mathura_1D_CNN_consumption_only_pilot",
        "device": device,
        "train_samples": len(X_tr_p),
        "batch_size": 256,
        "mean_epoch_duration_sec": round(float(np.mean(ind_epoch_times)), 4),
        "mean_throughput_samples_per_sec": round(float(np.mean(ind_throughput)), 1),
        "final_val_loss": round(float(ind_val_losses[-1]), 6),
        "epoch_durations": [round(t, 4) for t in ind_epoch_times],
        "val_losses": [round(l, 6) for l in ind_val_losses]
    }

    # Save pilot metrics to artifacts
    out_path = Path("artifacts/metrics/compute_pilot_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pilot_metrics, f, indent=2)
    print(f"\nSaved compute pilot benchmark metrics to {out_path}")
    return pilot_metrics


if __name__ == "__main__":
    run_compute_pilot()

