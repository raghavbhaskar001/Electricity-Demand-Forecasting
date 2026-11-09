"""
Phase 3.4 Consolidated Results Generator
Produces artifacts/metrics/phase_3_4_consolidated_results.csv
"""

import json
import pandas as pd
from pathlib import Path

def generate_consolidated_results():
    with open("artifacts/metrics/training_results.json", "r") as f:
        data = json.load(f)

    rows = []
    for d in data:
        exp_id = d["exp_id"]
        scope = d["cohort_type"]
        district = d["district"]
        model = d["model"]
        feature = d["feature_config"]
        in_feats = d["in_features"]
        best_ep = d["best_epoch"]
        tot_eps = d["total_epochs"]
        duration = d["training_duration_sec"]
        
        m = d["metrics"]
        rmse = m.get("rmse_kw")
        mae = m.get("mae_kw")
        wape = m.get("wape_pct")
        mape = m.get("mape_pct")
        r2 = m.get("r2_score")
        nrmse = m.get("nrmse_pct")
        ss_pers = m.get("persistence_skill_score")
        peak_mae = m.get("peak_mae_kw")
        peak_mape = m.get("peak_mape_pct")

        if scope == "aggregate":
            stat_status = "VALIDATED (Phase 3.2B Diebold-Mariano)"
        else:
            stat_status = "PENDING (Phase 3.3 Panel Statistical Testing)"

        rows.append({
            "experiment_id": exp_id,
            "scope": scope,
            "district": district,
            "model_architecture": model,
            "feature_configuration": feature,
            "input_features_count": in_feats,
            "best_epoch": best_ep,
            "total_epochs": tot_eps,
            "training_duration_sec": duration,
            "test_rmse_kw": rmse,
            "test_mae_kw": mae,
            "test_wape_pct": wape,
            "test_mape_pct": mape,
            "test_r2_score": r2,
            "test_nrmse_pct": nrmse,
            "persistence_skill_score": ss_pers,
            "test_peak_mae_kw": peak_mae,
            "test_peak_mape_pct": peak_mape,
            "statistical_validation_status": stat_status
        })

    df = pd.DataFrame(rows)
    out_path = Path("artifacts/metrics/phase_3_4_consolidated_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {out_path} with {len(df)} experiment rows.")
    return df

if __name__ == "__main__":
    generate_consolidated_results()

