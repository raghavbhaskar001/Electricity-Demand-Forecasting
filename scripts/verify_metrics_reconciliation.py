"""
Reconciliation Verification Script
Compares original Phase 3.0 logged metrics against Phase 3.2 regenerated predictions
"""

import json
import pandas as pd
import numpy as np
from src.evaluate import compute_forecasting_metrics

def main():
    with open("artifacts/metrics/training_results.json") as f:
        orig_results = {d["exp_id"]: d["metrics"] for d in json.load(f) if d["cohort_type"] == "aggregate"}

    df_preds = pd.read_parquet("artifacts/metrics/aggregate_test_predictions.parquet")

    print(f"Comparing all 18 aggregate models:")
    header = f"{'Experiment ID':<52} | {'Orig RMSE':<9} | {'Regen RMSE':<10} | {'RMSE Diff':<9} | {'Orig MAE':<8} | {'Regen MAE':<9} | {'MAE Diff':<8} | {'Status'}"
    print(header)
    print("-" * len(header))

    all_match = True
    for exp_id, orig_m in orig_results.items():
        sub = df_preds[df_preds["exp_id"] == exp_id]
        act = sub["actual_demand_kw"].values
        pred = sub["predicted_demand_kw"].values
        pers = sub["naive_persistence_kw"].values
        regen_m = compute_forecasting_metrics(act, pred, pers)
        
        diff_rmse = abs(orig_m["rmse_kw"] - regen_m["rmse_kw"])
        diff_mae = abs(orig_m["mae_kw"] - regen_m["mae_kw"])
        
        match = (diff_rmse < 1e-4) and (diff_mae < 1e-4)
        if not match:
            all_match = False
        status = "EXACT MATCH" if match else "MISMATCH"
        print(f"{exp_id:<52} | {orig_m['rmse_kw']:>9.4f} | {regen_m['rmse_kw']:>10.4f} | {diff_rmse:>9.5f} | {orig_m['mae_kw']:>8.4f} | {regen_m['mae_kw']:>9.4f} | {diff_mae:>8.5f} | {status}")

    print("-" * len(header))
    print("VERDICT: ALL 18 AGGREGATE REGENERATED METRICS MATCH ORIGINAL TRAINING LOGS EXACTLY:", all_match)

if __name__ == "__main__":
    main()

