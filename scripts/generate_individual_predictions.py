"""
Generate Timestamp-Level Test Predictions for Pooled Individual-Meter Models
Phase 3.2 - Prediction Artifact Generation for Colab GPU / Local Runtime
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import (
    extract_contiguous_sequences,
    temporal_train_val_test_split,
    FEATURE_SETS,
    load_meter_metadata
)
from src.train_runner import get_model, StandardScaler, PooledMeterTargetScaler, compute_meter_distribution
from src.evaluate import compute_forecasting_metrics

def generate_individual_predictions(
    checkpoints_dir: str = "artifacts/checkpoints",
    output_dir: str = "artifacts/metrics",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1024
):
    ckpt_dir = Path(checkpoints_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    districts = ["mathura", "bareilly"]
    models = ["1D_CNN", "LSTM", "CNN_LSTM"]
    feature_configs = {
        "consumption_only": FEATURE_SETS["consumption_only"],
        "consumption_weather_retro": FEATURE_SETS["consumption_weather_retro"],
        "consumption_weather_operational": FEATURE_SETS["consumption_weather_operational"]
    }
    
    print("=" * 80)
    print(f"POOLED INDIVIDUAL PREDICTION GENERATION | Device: {device}")
    print("=" * 80)

    # Check available checkpoints
    available_ckpts = list(ckpt_dir.glob("ind_*.pt"))
    print(f"Found {len(available_ckpts)} individual checkpoints in {ckpt_dir}")
    if len(available_ckpts) == 0:
        print("No individual checkpoints found. Run this in Google Colab where checkpoints are stored!")
        return None

    all_dfs = []
    
    for district in districts:
        cons_file = f"data/processed/{district}_consumption_clean.parquet"
        weather_file = f"data/processed/{district}_weather_clean.parquet"
        
        if not Path(cons_file).exists() or not Path(weather_file).exists():
            print(f"Skipping {district}: clean data missing.")
            continue
            
        df_cons = pd.read_parquet(cons_file)
        df_weather = pd.read_parquet(weather_file)
        
        # Load meter metadata
        meta = load_meter_metadata(district)
        meters = meta.index.tolist() if not meta.empty else df_cons.columns.tolist()
        meters = [m for m in meters if m in df_cons.columns]
        
        # Pre-align weather
        weather_cols_retro = [c for c in df_weather.columns if not c.endswith("_operational")]
        weather_cols_op = [c for c in df_weather.columns if c.endswith("_operational")]
        
        for f_name, f_cols in feature_configs.items():
            print(f"\n--- {district.upper()} Individual Pooled | {f_name} ---")
            
            # Check if any model for this district & feature config has a checkpoint
            needed_models = [m for m in models if (ckpt_dir / f"ind_{district}_{m}_{f_name}.pt").exists()]
            if not needed_models:
                print(f"  No checkpoints found for {district} {f_name}. Skipping.")
                continue
                
            pool_X_tr, pool_y_tr = [], []
            pool_X_te, pool_y_te = [], []
            pool_naive_te = []
            pool_ts_te = []
            pool_meter_te = []
            meter_test_slices = []
            current_te_idx = 0
            
            sub_f_cols = [c for c in f_cols if c != "demand_target"]
            
            for m in meters:
                s_cons = df_cons[[m]].rename(columns={m: "demand_target"})
                if f_name == "consumption_only":
                    df_m = s_cons
                elif f_name == "consumption_weather_retro":
                    df_m = s_cons.join(df_weather[weather_cols_retro], how="inner")
                else:
                    df_m = s_cons.join(df_weather[weather_cols_op], how="inner")
                    
                df_m = df_m.dropna()
                if len(df_m) < 100:
                    continue
                    
                X_m, y_m, ts_m = extract_contiguous_sequences(df_m, "demand_target", f_cols, 16, 1)
                (X_tr_m, y_tr_m, _), (X_va_m, y_va_m, _), (X_te_m, y_te_m, ts_te_m) = temporal_train_val_test_split(
                    X_m, y_m, ts_m, 0.70, 0.15, 16
                )
                
                if len(X_tr_m) == 0 or len(X_te_m) == 0:
                    continue
                    
                m_feat_scaler = StandardScaler()
                m_targ_scaler = StandardScaler()
                
                X_tr_m_s = m_feat_scaler.fit_transform(X_tr_m.reshape(-1, len(sub_f_cols))).reshape(X_tr_m.shape)
                X_te_m_s = m_feat_scaler.transform(X_te_m.reshape(-1, len(sub_f_cols))).reshape(X_te_m.shape)
                y_tr_m_s = m_targ_scaler.fit_transform(y_tr_m)
                
                pool_X_te.append(X_te_m_s)
                pool_y_te.append(y_te_m) # raw kW
                pool_naive_te.append(X_te_m[:, -1, 0:1]) # raw kW
                pool_ts_te.append(ts_te_m)
                pool_meter_te.extend([m] * len(y_te_m))
                
                te_len = len(y_te_m)
                meter_test_slices.append((m, current_te_idx, current_te_idx + te_len, m_targ_scaler))
                current_te_idx += te_len
                
            X_te_pool = np.vstack(pool_X_te)
            y_te_pool = np.vstack(pool_y_te).flatten()
            y_naive_pool = np.vstack(pool_naive_te).flatten()
            ts_pool = np.concatenate(pool_ts_te)
            meter_ids_pool = np.array(pool_meter_te)
            pooled_scaler = PooledMeterTargetScaler(meter_test_slices)
            
            X_te_tensor = torch.from_numpy(X_te_pool).float()
            
            for m_name in needed_models:
                exp_id = f"ind_{district}_{m_name}_{f_name}"
                ckpt_path = ckpt_dir / f"{exp_id}.pt"
                
                model = get_model(m_name, len(f_cols), sequence_length=16).to(device)
                chk = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(chk["model_state_dict"])
                model.eval()
                
                preds_scaled = []
                with torch.no_grad():
                    for b_start in range(0, len(X_te_tensor), batch_size):
                        b_x = X_te_tensor[b_start:b_start + batch_size].to(device)
                        preds_scaled.append(model(b_x).cpu().numpy())
                preds_scaled = np.vstack(preds_scaled)
                preds_kw = pooled_scaler.inverse_transform(preds_scaled).flatten()
                
                met = compute_forecasting_metrics(y_te_pool, preds_kw, y_naive_pool)
                print(f"    {exp_id}: RMSE={met['rmse_kw']} kW, R2={met['r2_score']}, SS_pers={met['persistence_skill_score']}")
                
                df_pred = pd.DataFrame({
                    "timestamp": pd.to_datetime(ts_pool),
                    "cohort_type": "individual_pooled",
                    "district": district,
                    "meter_id": meter_ids_pool,
                    "actual_demand_kw": y_te_pool,
                    "predicted_demand_kw": preds_kw,
                    "naive_persistence_kw": y_naive_pool,
                    "model_architecture": m_name,
                    "feature_config": f_name,
                    "exp_id": exp_id
                })
                all_dfs.append(df_pred)
                
    if all_dfs:
        res_df = pd.concat(all_dfs, ignore_index=True)
        out_parquet = out_dir / "individual_test_predictions.parquet"
        res_df.to_parquet(out_parquet, index=False)
        print(f"\nSaved {len(res_df)} individual test predictions to {out_parquet}")
        return res_df
    return None

if __name__ == "__main__":
    generate_individual_predictions()

