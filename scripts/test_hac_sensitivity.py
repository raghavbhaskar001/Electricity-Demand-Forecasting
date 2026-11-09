"""
HAC Truncation Lag Sensitivity Analysis
Evaluates stability of Diebold-Mariano tests across K in {4, 8, 16, 96}
"""

import pandas as pd
import numpy as np
from src.statistical_tests import diebold_mariano_test, holm_bonferroni_correction

def run_sensitivity():
    df_preds = pd.read_parquet("artifacts/metrics/aggregate_test_predictions.parquet")
    lags = [4, 8, 16, 96]
    
    districts = ["mathura", "bareilly"]
    all_rows = []

    for district in districts:
        df_dist = df_preds[df_preds["district"] == district]
        ref = df_dist[(df_dist["model_architecture"] == "1D_CNN") & (df_dist["feature_config"] == "consumption_only")].sort_values("timestamp")
        y_true = ref["actual_demand_kw"].values
        y_pers = ref["naive_persistence_kw"].values
        
        pred_dict = {}
        for (m, f), grp in df_dist.groupby(["model_architecture", "feature_config"]):
            pred_dict[(m, f)] = grp.sort_values("timestamp")["predicted_demand_kw"].values

        best_m = "LSTM" if district == "mathura" else "1D_CNN"
        
        hypotheses = [
            ("1D_CNN vs LSTM", pred_dict[("1D_CNN", "consumption_only")], pred_dict[("LSTM", "consumption_only")]),
            ("LSTM vs CNN-LSTM", pred_dict[("LSTM", "consumption_only")], pred_dict[("CNN_LSTM", "consumption_only")]),
            ("1D_CNN vs CNN-LSTM", pred_dict[("1D_CNN", "consumption_only")], pred_dict[("CNN_LSTM", "consumption_only")]),
            ("Consumption vs Retro", pred_dict[(best_m, "consumption_only")], pred_dict[(best_m, "consumption_weather_retro")]),
            ("Consumption vs Operational", pred_dict[(best_m, "consumption_only")], pred_dict[(best_m, "consumption_weather_operational")]),
            ("Retro vs Operational", pred_dict[(best_m, "consumption_weather_retro")], pred_dict[(best_m, "consumption_weather_operational")]),
            ("Best DL vs Persistence", pred_dict[(best_m, "consumption_only")], y_pers),
        ]

        for hyp_name, p1, p2 in hypotheses:
            row = {"district": district, "hypothesis": hyp_name}
            for k in lags:
                res = diebold_mariano_test(y_true, p1, p2, loss_type="squared_error", max_lags=k)
                row[f"DM_K{k}"] = res["dm_stat"]
                row[f"p_raw_K{k}"] = res["p_value_raw"]
                row[f"sig_K{k}"] = res["p_value_raw"] < 0.05
            all_rows.append(row)

    df_sens = pd.DataFrame(all_rows)
    print("=" * 100)
    print("HAC TRUNCATION LAG SENSITIVITY (K = 4, 8, 16, 96)")
    print("=" * 100)
    for _, r in df_sens.iterrows():
        print(f"{r['district'].upper():<9} | {r['hypothesis']:<25} | K=4: DM={r['DM_K4']:>7.3f} (p={r['p_raw_K4']:.2e}) | K=8: DM={r['DM_K8']:>7.3f} | K=16: DM={r['DM_K16']:>7.3f} | K=96: DM={r['DM_K96']:>7.3f} (p={r['p_raw_K96']:.2e})")

    # Check stability
    stable_direction = True
    stable_significance = True
    for _, r in df_sens.iterrows():
        stats = [r["DM_K4"], r["DM_K8"], r["DM_K16"], r["DM_K96"]]
        sigs = [r["sig_K4"], r["sig_K8"], r["sig_K16"], r["sig_K96"]]
        if not (all(s > 0 for s in stats) or all(s < 0 for s in stats)):
            stable_direction = False
        if not (all(sigs) or not any(sigs)):
            stable_significance = False

    print("=" * 100)
    print("DIRECTION STABILITY ACROSS ALL LAGS (K=4, 8, 16, 96):", stable_direction)
    print("SIGNIFICANCE STABILITY ACROSS ALL LAGS (K=4, 8, 16, 96):", stable_significance)
    print("=" * 100)
    
    df_sens.to_csv("artifacts/metrics/hac_lag_sensitivity.csv", index=False)
    return df_sens

if __name__ == "__main__":
    run_sensitivity()

