"""
Master Training Runner for Short-Term Electricity Demand Forecasting
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models

Executes the approved 36-experiment matrix:
- 18 Aggregate Experiments (Mathura Aggregate: 9, Bareilly Aggregate: 9)
- 18 Individual Meter Pooled Experiments (Mathura Individual: 9, Bareilly Individual: 9)
"""

import os
import sys
import time
import json
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.dataset import (
    FEATURE_SETS,
    extract_contiguous_sequences,
    temporal_train_val_test_split,
)
from src.evaluate import compute_forecasting_metrics

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from src.models import CNN1DForecaster, LSTMForecaster, CNNLSTMForecaster
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class FastTensorDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.as_tensor(X, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


class PooledMeterTargetScaler:
    """
    Inverts pooled individual-meter predictions by applying each meter's
    specific training StandardScaler (mean, std) to its corresponding test partition slice.
    Guarantees that test evaluation occurs strictly in physical kW units.
    """
    def __init__(self, meter_slices: List[Tuple[str, int, int, Any]]):
        self.meter_slices = meter_slices

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        y_kw = np.zeros_like(y_scaled, dtype=np.float64)
        for meter_id, start_idx, end_idx, scaler in self.meter_slices:
            if start_idx < len(y_scaled):
                sub_scaled = y_scaled[start_idx:min(end_idx, len(y_scaled))]
                y_kw[start_idx:min(end_idx, len(y_scaled))] = scaler.inverse_transform(sub_scaled)
        return y_kw


def compute_meter_distribution(
    actuals_kw: np.ndarray,
    preds_kw: np.ndarray,
    y_naive_te: np.ndarray,
    meter_slices: List[Tuple[str, int, int, Any]]
) -> Dict[str, Any]:
    """
    Evaluates forecast accuracy independently per household smart meter
    and returns median, IQR, min, and max summary distributions.
    """
    m_rmse, m_mae, m_wape, m_r2, m_sspers = [], [], [], [], []
    meter_records = {}

    for m_id, start, end, _ in meter_slices:
        sub_act = actuals_kw[start:end]
        sub_pred = preds_kw[start:end]
        sub_pers = y_naive_te[start:end]
        if len(sub_act) > 0:
            met = compute_forecasting_metrics(sub_act, sub_pred, sub_pers)
            meter_records[m_id] = met
            if met["rmse_kw"] is not None:
                m_rmse.append(met["rmse_kw"])
            if met["mae_kw"] is not None:
                m_mae.append(met["mae_kw"])
            if met["wape_pct"] is not None:
                m_wape.append(met["wape_pct"])
            if met["r2_score"] is not None:
                m_r2.append(met["r2_score"])
            if met["persistence_skill_score"] is not None:
                m_sspers.append(met["persistence_skill_score"])

    return {
        "meter_count": len(meter_slices),
        "median_rmse_kw": round(float(np.median(m_rmse)), 4) if m_rmse else None,
        "iqr_rmse_kw": round(float(np.percentile(m_rmse, 75) - np.percentile(m_rmse, 25)), 4) if m_rmse else None,
        "min_rmse_kw": round(float(np.min(m_rmse)), 4) if m_rmse else None,
        "max_rmse_kw": round(float(np.max(m_rmse)), 4) if m_rmse else None,
        "median_mae_kw": round(float(np.median(m_mae)), 4) if m_mae else None,
        "median_wape_pct": round(float(np.median(m_wape)), 2) if m_wape else None,
        "median_r2_score": round(float(np.median(m_r2)), 4) if m_r2 else None,
        "median_ss_pers": round(float(np.median(m_sspers)), 4) if m_sspers else None,
        "per_meter_metrics": meter_records
    }




def get_model(architecture: str, in_features: int, sequence_length: int = 16):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model construction.")
    if architecture == "1D_CNN":
        return CNN1DForecaster(in_features=in_features, sequence_length=sequence_length)
    elif architecture == "LSTM":
        return LSTMForecaster(in_features=in_features)
    elif architecture == "CNN_LSTM":
        return CNNLSTMForecaster(in_features=in_features)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    count = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(batch_y)
        count += len(batch_y)
    return total_loss / max(count, 1)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            total_loss += loss.item() * len(batch_y)
            count += len(batch_y)
    return total_loss / max(count, 1)


def train_and_evaluate_experiment(
    exp_id: str,
    model_name: str,
    feature_config_name: str,
    cohort_type: str,  # 'aggregate' or 'individual_pooled'
    district: str,      # 'mathura' or 'bareilly'
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    target_scaler: Any,
    y_naive_te: np.ndarray,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 8,
    checkpoints_dir: str = "artifacts/checkpoints",
    logs_dir: str = "artifacts/metrics/training_logs",
    device: str = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Standard training, early stopping, checkpointing, and test evaluation pipeline.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training execution.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ckpt_path = Path(checkpoints_dir) / f"{exp_id}.pt"
    log_path = Path(logs_dir) / f"{exp_id}.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Upfront Completion Guard (Check if experiment is already fully completed and verified)
    if ckpt_path.exists() and log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                saved_log = json.load(f)
            if "record" in saved_log and saved_log["record"].get("metrics", {}).get("rmse_kw") is not None:
                if ckpt_path.stat().st_size > 100000:
                    print(f"[COMPLETION GUARD] Experiment {exp_id} already verified complete. Loading cached record.")
                    return saved_log["record"]
        except Exception as e:
            print(f"[COMPLETION GUARD] Experiment {exp_id} has invalid/partial log ({e}). Proceeding to clean run.")


    in_features = X_tr.shape[-1]
    model = get_model(model_name, in_features=in_features).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)

    tr_dataset = FastTensorDataset(X_tr, y_tr)
    va_dataset = FastTensorDataset(X_va, y_va)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device=="cuda"))
    va_loader = DataLoader(va_dataset, batch_size=batch_size * 2, shuffle=False, pin_memory=(device=="cuda"))

    history = {"train_loss": [], "val_loss": [], "lr": [], "epoch_times": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None
    patience_counter = 0

    t_start = time.time()
    for ep in range(1, epochs + 1):
        t_ep0 = time.time()
        tr_loss = train_epoch(model, tr_loader, optimizer, criterion, device)
        va_loss = evaluate_loss(model, va_loader, criterion, device)
        scheduler.step(va_loss)
        curr_lr = optimizer.param_groups[0]["lr"]
        ep_duration = time.time() - t_ep0

        history["train_loss"].append(round(tr_loss, 6))
        history["val_loss"].append(round(va_loss, 6))
        history["lr"].append(curr_lr)
        history["epoch_times"].append(round(ep_duration, 2))

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch = ep
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save({
                "epoch": ep,
                "model_state_dict": best_weights,
                "val_loss": best_val_loss,
                "optimizer_state_dict": optimizer.state_dict(),
                "exp_id": exp_id
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    total_duration = time.time() - t_start

    # Restore best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)
    elif ckpt_path.exists():
        chk = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(chk["model_state_dict"])

    # Test Evaluation
    model.eval()
    te_dataset = FastTensorDataset(X_te, y_te)
    te_loader = DataLoader(te_dataset, batch_size=batch_size * 2, shuffle=False)
    preds_scaled = []
    with torch.no_grad():
        for batch_x, _ in te_loader:
            batch_x = batch_x.to(device)
            p = model(batch_x).cpu().numpy()
            preds_scaled.append(p)
    preds_scaled = np.vstack(preds_scaled)

    # Invert target scaling to evaluate in physical kW
    if hasattr(target_scaler, "inverse_transform"):
        preds_kw = target_scaler.inverse_transform(preds_scaled)
        actuals_kw = target_scaler.inverse_transform(y_te)
    else:
        preds_kw = preds_scaled
        actuals_kw = y_te

    test_metrics = compute_forecasting_metrics(
        y_true=actuals_kw,
        y_pred=preds_kw,
        y_persistence=y_naive_te
    )

    meter_dist = None
    if hasattr(target_scaler, "meter_slices"):
        meter_dist = compute_meter_distribution(actuals_kw, preds_kw, y_naive_te, target_scaler.meter_slices)

    result_record = {
        "exp_id": exp_id,
        "cohort_type": cohort_type,
        "district": district,
        "model": model_name,
        "feature_config": feature_config_name,
        "in_features": in_features,
        "best_epoch": best_epoch,
        "total_epochs": len(history["train_loss"]),
        "training_duration_sec": round(total_duration, 2),
        "best_val_loss": round(best_val_loss, 6),
        "metrics": test_metrics,
        "meter_distribution": meter_dist,
        "checkpoint_path": str(ckpt_path)
    }

    # Save training log
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "record": result_record,
            "history": history
        }, f, indent=2)

    return result_record


def run_master_training(
    target_scope: str = "all",  # "all", "aggregate", "individual"
    districts: List[str] = ["mathura", "bareilly"],
    epochs_agg: int = 40,
    epochs_ind: int = 15,
    batch_size_agg: int = 64,
    batch_size_ind: int = 512,
    checkpoints_dir: str = "artifacts/checkpoints",
    metrics_dir: str = "artifacts/metrics",
    device: str = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
) -> List[Dict[str, Any]]:
    """
    Executes the approved master training matrix and records all benchmark metrics.
    """
    if not TORCH_AVAILABLE:
        print("WARNING: PyTorch not installed in this environment. Script ready for Google Colab GPU runtime.")
        return []

    print(f"\n{'='*80}\nSTARTING MASTER DEEP LEARNING TRAINING MATRIX\nTarget Scope: {target_scope} | Device: {device}\n{'='*80}")
    
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    logs_dir = Path(metrics_dir) / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    models_to_train = ["1D_CNN", "LSTM", "CNN_LSTM"]
    feature_configs = {
        "consumption_only": FEATURE_SETS["consumption_only"],
        "consumption_weather_retro": FEATURE_SETS["consumption_weather_retro"],
        "consumption_weather_operational": FEATURE_SETS["consumption_weather_operational"]
    }

    # --------------------------------------------------------------------------
    # 1. AGGREGATE COHORT EXPERIMENTS (18 total)
    # --------------------------------------------------------------------------
    if target_scope in ["all", "aggregate"]:
        for district in districts:
            feat_file = f"data/processed/{district}_features.parquet"
            if not Path(feat_file).exists():
                print(f"Skipping {district} aggregate: {feat_file} not found.")
                continue

            print(f"\n>>> Loading Aggregate Data for {district.upper()} ...")
            df_feat = pd.read_parquet(feat_file)
            target_col = "aggregate_demand_kw_primary"

            for f_name, f_cols in feature_configs.items():
                print(f"\n--- Aggregate {district.upper()} | Feature Config: {f_name} (in_feat={len(f_cols)}) ---")
                X, y, ts = extract_contiguous_sequences(df_feat, target_col, f_cols, lookback_len=16, horizon_len=1)
                
                # Split with 16-step boundary purge gap
                (X_tr, y_tr, ts_tr), (X_va, y_va, ts_va), (X_te, y_te, ts_te) = temporal_train_val_test_split(
                    X, y, ts, train_ratio=0.70, val_ratio=0.15, purge_gap=16
                )
                print(f"Partitions: Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)}")

                # Persistence on test in kW
                y_naive_te = X_te[:, -1, 0:1] # raw kW

                # Standard scaling fit on train only
                n_feats = len(f_cols)
                feat_scaler = StandardScaler()
                targ_scaler = StandardScaler()

                X_tr_s = feat_scaler.fit_transform(X_tr.reshape(-1, n_feats)).reshape(X_tr.shape)
                X_va_s = feat_scaler.transform(X_va.reshape(-1, n_feats)).reshape(X_va.shape)
                X_te_s = feat_scaler.transform(X_te.reshape(-1, n_feats)).reshape(X_te.shape)

                y_tr_s = targ_scaler.fit_transform(y_tr)
                y_va_s = targ_scaler.transform(y_va)
                y_te_s = targ_scaler.transform(y_te)

                for m_name in models_to_train:
                    exp_id = f"agg_{district}_{m_name}_{f_name}"
                    print(f"Training experiment {exp_id} ...", end=" ")
                    sys.stdout.flush()

                    rec = train_and_evaluate_experiment(
                        exp_id=exp_id,
                        model_name=m_name,
                        feature_config_name=f_name,
                        cohort_type="aggregate",
                        district=district,
                        X_tr=X_tr_s, y_tr=y_tr_s,
                        X_va=X_va_s, y_va=y_va_s,
                        X_te=X_te_s, y_te=y_te_s,
                        target_scaler=targ_scaler,
                        y_naive_te=y_naive_te,
                        batch_size=batch_size_agg,
                        epochs=epochs_agg,
                        patience=8,
                        checkpoints_dir=checkpoints_dir,
                        logs_dir=str(logs_dir),
                        device=device
                    )
                    results.append(rec)
                    m = rec["metrics"]
                    print(f"Done in {rec['training_duration_sec']}s (Best Ep: {rec['best_epoch']}) | RMSE: {m['rmse_kw']} kW | WAPE: {m['wape_pct']}% | R2: {m['r2_score']} | SS_pers: {m['persistence_skill_score']}")

    # --------------------------------------------------------------------------
    # 2. INDIVIDUAL METER POOLED EXPERIMENTS (18 total)
    # --------------------------------------------------------------------------
    if target_scope in ["all", "individual"]:
        for district in districts:
            ind_file = f"data/processed/{district}_individual_meters.parquet"
            feat_file = f"data/processed/{district}_features.parquet"
            if not Path(ind_file).exists() or not Path(feat_file).exists():
                print(f"Skipping {district} individual: required files not found.")
                continue

            print(f"\n>>> Preparing Pooled Individual Meter Data for {district.upper()} ...")
            df_ind = pd.read_parquet(ind_file)
            df_feat = pd.read_parquet(feat_file)
            meter_cols = [c for c in df_ind.columns if c != "timestamp"]
            print(f"Found {len(meter_cols)} meters in {district} cohort.")

            # Merge weather covariates
            weather_cols = [c for c in df_feat.columns if c.startswith("retro_") or c.startswith("operational_")]
            df_merged = pd.merge(df_ind, df_feat[["timestamp"] + weather_cols], on="timestamp", how="inner")

            for f_name, f_cols in feature_configs.items():
                print(f"\n--- Individual Pooled {district.upper()} | Feature Config: {f_name} ---")
                
                # Extract and scale sequences per meter to preserve meter baseline
                pool_X_tr, pool_y_tr = [], []
                pool_X_va, pool_y_va = [], []
                pool_X_te, pool_y_te = [], []
                pool_naive_te = []
                meter_test_slices = []
                current_te_idx = 0

                for m in meter_cols:
                    # Select target meter and required weather features
                    sub_f_cols = ["demand_target"] + [c for c in f_cols if c != "demand_target"]
                    sub_df = df_merged[["timestamp", m] + [c for c in f_cols if c != "demand_target"]].copy()
                    sub_df["demand_target"] = sub_df[m]

                    X_m, y_m, ts_m = extract_contiguous_sequences(sub_df, "demand_target", sub_f_cols, lookback_len=16, horizon_len=1)
                    if len(X_m) == 0:
                        continue

                    # Split
                    (X_tr_m, y_tr_m, _), (X_va_m, y_va_m, _), (X_te_m, y_te_m, _) = temporal_train_val_test_split(
                        X_m, y_m, ts_m, train_ratio=0.70, val_ratio=0.15, purge_gap=16
                    )

                    if len(X_tr_m) == 0 or len(X_te_m) == 0:
                        continue

                    # Per-meter scaling fit strictly on this meter's training split
                    m_feat_scaler = StandardScaler()
                    m_targ_scaler = StandardScaler()

                    X_tr_m_s = m_feat_scaler.fit_transform(X_tr_m.reshape(-1, len(sub_f_cols))).reshape(X_tr_m.shape)
                    X_va_m_s = m_feat_scaler.transform(X_va_m.reshape(-1, len(sub_f_cols))).reshape(X_va_m.shape)
                    X_te_m_s = m_feat_scaler.transform(X_te_m.reshape(-1, len(sub_f_cols))).reshape(X_te_m.shape)

                    y_tr_m_s = m_targ_scaler.fit_transform(y_tr_m)
                    y_va_m_s = m_targ_scaler.transform(y_va_m)
                    y_te_m_s = m_targ_scaler.transform(y_te_m)

                    pool_X_tr.append(X_tr_m_s)
                    pool_y_tr.append(y_tr_m_s)
                    pool_X_va.append(X_va_m_s)
                    pool_y_va.append(y_va_m_s)
                    pool_X_te.append(X_te_m_s)
                    pool_y_te.append(y_te_m_s)
                    pool_naive_te.append(X_te_m[:, -1, 0:1]) # raw kW

                    te_len = len(y_te_m)
                    meter_test_slices.append((m, current_te_idx, current_te_idx + te_len, m_targ_scaler))
                    current_te_idx += te_len

                # Stack pooled tensors
                X_tr_pool = np.vstack(pool_X_tr)
                y_tr_pool = np.vstack(pool_y_tr)
                X_va_pool = np.vstack(pool_X_va)
                y_va_pool = np.vstack(pool_y_va)
                X_te_pool = np.vstack(pool_X_te)
                y_te_pool = np.vstack(pool_y_te)
                y_naive_pool = np.vstack(pool_naive_te)
                pooled_target_scaler = PooledMeterTargetScaler(meter_test_slices)
                print(f"Pooled Census -> Train: {len(X_tr_pool)}, Val: {len(X_va_pool)}, Test: {len(X_te_pool)}")

                for m_name in models_to_train:
                    exp_id = f"ind_{district}_{m_name}_{f_name}"
                    print(f"Training pooled experiment {exp_id} ...", end=" ")
                    sys.stdout.flush()

                    rec = train_and_evaluate_experiment(
                        exp_id=exp_id,
                        model_name=m_name,
                        feature_config_name=f_name,
                        cohort_type="individual_pooled",
                        district=district,
                        X_tr=X_tr_pool, y_tr=y_tr_pool,
                        X_va=X_va_pool, y_va=y_va_pool,
                        X_te=X_te_pool, y_te=y_te_pool,
                        target_scaler=pooled_target_scaler,
                        y_naive_te=y_naive_pool,
                        batch_size=batch_size_ind,
                        epochs=epochs_ind,
                        patience=6,
                        checkpoints_dir=checkpoints_dir,
                        logs_dir=str(logs_dir),
                        device=device
                    )
                    results.append(rec)
                    m = rec["metrics"]
                    print(f"Done in {rec['training_duration_sec']}s (Best Ep: {rec['best_epoch']}) | RMSE: {m['rmse_kw']} kW | WAPE: {m['wape_pct']}% | R2: {m['r2_score']} | SS_pers: {m['persistence_skill_score']}")

    # --------------------------------------------------------------------------
    # 3. EXPORT CONSOLIDATED BENCHMARK SUMMARY
    # --------------------------------------------------------------------------
    if results:
        results_json_path = Path(metrics_dir) / "training_results.json"
        existing_records = []
        if results_json_path.exists():
            try:
                with open(results_json_path, "r", encoding="utf-8") as f:
                    existing_records = json.load(f)
            except Exception:
                existing_records = []

        # Merge records by exp_id
        rec_map = {r["exp_id"]: r for r in existing_records}
        for r in results:
            rec_map[r["exp_id"]] = r
        merged_records = list(rec_map.values())

        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(merged_records, f, indent=2)

        summary_rows = []
        for r in merged_records:
            m = r["metrics"]
            mdist = r.get("meter_distribution") or {}
            summary_rows.append({
                "Experiment_ID": r["exp_id"],
                "Cohort_Type": r["cohort_type"],
                "District": r["district"],
                "Model": r["model"],
                "Feature_Config": r["feature_config"],
                "In_Features": r["in_features"],
                "Best_Epoch": r["best_epoch"],
                "Total_Epochs": r["total_epochs"],
                "Training_Time_s": r["training_duration_sec"],
                "RMSE_kW": m["rmse_kw"],
                "MAE_kW": m["mae_kw"],
                "WAPE_pct": m["wape_pct"],
                "MAPE_pct": m["mape_pct"],
                "R2_Score": m["r2_score"],
                "nRMSE_pct": m.get("nrmse_pct", m.get("nRMSE_pct")),
                "Persistence_Skill_Score": m["persistence_skill_score"],
                "Peak_MAE_kW": m["peak_mae_kw"],
                "Peak_MAPE_pct": m["peak_mape_pct"],
                "Median_RMSE_kW": mdist.get("median_rmse_kw"),
                "IQR_RMSE_kW": mdist.get("iqr_rmse_kw"),
                "Median_R2": mdist.get("median_r2_score"),
                "Median_SS_pers": mdist.get("median_ss_pers")
            })
        df_summary = pd.DataFrame(summary_rows)
        csv_path = Path(metrics_dir) / "experiment_summary.csv"
        df_summary.to_csv(csv_path, index=False)
        print(f"\nSuccessfully wrote {len(merged_records)} experiment benchmarks to {csv_path} and {results_json_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Master Deep Learning Training Runner")
    parser.add_argument("--scope", type=str, default="all", choices=["all", "aggregate", "individual"])
    parser.add_argument("--districts", nargs="+", default=["mathura", "bareilly"])
    parser.add_argument("--device", type=str, default="cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
    args = parser.parse_args()

    run_master_training(target_scope=args.scope, districts=args.districts, device=args.device)
