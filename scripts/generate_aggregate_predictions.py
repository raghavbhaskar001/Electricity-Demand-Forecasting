"""
Generate Timestamp-Level Test Predictions for Aggregate Cohort Models
Phase 3.2 - Prediction Artifact Generation (Without Retraining)
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import extract_contiguous_sequences, temporal_train_val_test_split, FEATURE_SETS
from src.train_runner import get_model, StandardScaler
from src.evaluate import compute_forecasting_metrics

def generate_aggregate_predictions():
    districts = ["mathura", "bareilly"]
    models = ["1D_CNN", "LSTM", "CNN_LSTM"]
    feature_configs = {
        "consumption_only": FEATURE_SETS["consumption_only"],
        "consumption_weather_retro": FEATURE_SETS["consumption_weather_retro"],
        "consumption_weather_operational": FEATURE_SETS["consumption_weather_operational"]
    }
    
    ckpt_dir = Path("artifacts/checkpoints")
    out_dir = Path("artifacts/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_dfs = []
    summary_verifications = []

    print("=" * 80)
    print("REGENERATING TIMESTAMP-LEVEL TEST PREDICTIONS (AGGREGATE MODELS)")
    print("=" * 80)

    for district in districts:
        feat_path = Path(f"data/processed/{district}_features.parquet")
        if not feat_path.exists():
            print(f"Error: {feat_path} does not exist!")
            continue
            
        print(f"\nProcessing {district.upper()} aggregate cohort...")
        df_feat = pd.read_parquet(feat_path)
        target_col = "aggregate_demand_kw_primary"
        
        for f_name, f_cols in feature_configs.items():
            print(f"  Feature Config: {f_name} (in_feat={len(f_cols)})")
            
            # Extract identical contiguous sequences
            X, y, ts = extract_contiguous_sequences(
                df_feat, target_col, f_cols, lookback_len=16, horizon_len=1
            )
            
            # Strict chronological 70/15/15 temporal split with 16-step purge gap
            (X_tr, y_tr, ts_tr), (X_va, y_va, ts_va), (X_te, y_te, ts_te) = temporal_train_val_test_split(
                X, y, ts, train_ratio=0.70, val_ratio=0.15, purge_gap=16
            )
            
            y_naive_te = X_te[:, -1, 0:1] # raw physical kW persistence
            
            # Scalers fit strictly on train split
            n_feats = len(f_cols)
            feat_scaler = StandardScaler()
            targ_scaler = StandardScaler()
            
            X_tr_s = feat_scaler.fit_transform(X_tr.reshape(-1, n_feats)).reshape(X_tr.shape)
            X_te_s = feat_scaler.transform(X_te.reshape(-1, n_feats)).reshape(X_te.shape)
            y_tr_s = targ_scaler.fit_transform(y_tr)
            
            X_te_tensor = torch.from_numpy(X_te_s).float()
            
            for m_name in models:
                exp_id = f"agg_{district}_{m_name}_{f_name}"
                ckpt_path = ckpt_dir / f"{exp_id}.pt"
                
                if not ckpt_path.exists():
                    print(f"    WARNING: Checkpoint {ckpt_path} missing!")
                    continue
                    
                model = get_model(m_name, len(f_cols), sequence_length=16)
                chk = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(chk["model_state_dict"])
                model.eval()
                
                with torch.no_grad():
                    pred_scaled = model(X_te_tensor).numpy()
                
                preds_kw = targ_scaler.inverse_transform(pred_scaled).flatten()
                actuals_kw = y_te.flatten()
                persistence_kw = y_naive_te.flatten()
                
                metrics = compute_forecasting_metrics(actuals_kw, preds_kw, persistence_kw)
                print(f"    {exp_id}: RMSE={metrics['rmse_kw']} kW, R2={metrics['r2_score']}")
                
                summary_verifications.append({
                    "exp_id": exp_id,
                    "district": district,
                    "model": m_name,
                    "feature_config": f_name,
                    "test_rmse_kw": metrics["rmse_kw"],
                    "test_r2": metrics["r2_score"],
                    "test_mae_kw": metrics["mae_kw"],
                    "n_test_samples": len(actuals_kw)
                })
                
                df_pred = pd.DataFrame({
                    "timestamp": pd.to_datetime(ts_te),
                    "cohort_type": "aggregate",
                    "district": district,
                    "meter_id": "feeder_primary",
                    "actual_demand_kw": actuals_kw,
                    "predicted_demand_kw": preds_kw,
                    "naive_persistence_kw": persistence_kw,
                    "model_architecture": m_name,
                    "feature_config": f_name,
                    "exp_id": exp_id
                })
                all_dfs.append(df_pred)
                
    master_df = pd.concat(all_dfs, ignore_index=True)
    out_parquet = out_dir / "aggregate_test_predictions.parquet"
    out_csv = out_dir / "aggregate_test_predictions.csv"
    master_df.to_parquet(out_parquet, index=False)
    master_df.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 80)
    print(f"Successfully generated {len(master_df)} prediction records across 18 aggregate experiments.")
    print(f"Saved to {out_parquet} and {out_csv}")
    print("=" * 80)
    
    return master_df, summary_verifications

if __name__ == "__main__":
    generate_aggregate_predictions()

