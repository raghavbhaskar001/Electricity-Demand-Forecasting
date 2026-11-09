"""
Individual Smart Meter Panel Statistical Evaluation Runner
Phase 3.3 - Formal Meter-Level Wilcoxon Tests & Clustered Bootstrap
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.statistical_tests import holm_bonferroni_correction
from src.panel_statistical_tests import (
    compute_per_meter_loss_differentials,
    run_meter_wilcoxon_test,
    run_meter_blocked_bootstrap
)

def run_panel_evaluation():
    pred_path = Path("artifacts/metrics/individual_test_predictions.parquet")
    out_dir = Path("artifacts/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not pred_path.exists():
        print(f"Notice: {pred_path} does not exist yet.")
        print("To generate it, please import ind_checkpoints.zip into artifacts/checkpoints/ or run generate_individual_predictions.")
        return None

    print("=" * 80)
    print("RUNNING PANEL STATISTICAL EVALUATION (METER-LEVEL PAIRED TESTS)")
    print("=" * 80)

    df_preds = pd.read_parquet(pred_path)
    districts = ["mathura", "bareilly"]
    models = ["1D_CNN", "LSTM", "CNN_LSTM"]
    features = ["consumption_only", "consumption_weather_retro", "consumption_weather_operational"]

    all_test_records = []
    meter_stat_records = []

    for district in districts:
        df_dist = df_preds[df_preds["district"] == district]
        unique_meters = df_dist["meter_id"].unique()
        print(f"\nProcessing {district.upper()} panel ({len(unique_meters)} meters)...")

        # Compute meter-wise individual metrics table
        for exp_id in df_dist["exp_id"].unique():
            sub = df_dist[df_dist["exp_id"] == exp_id]
            for m_id, grp in sub.groupby("meter_id"):
                e = grp["actual_demand_kw"] - grp["predicted_demand_kw"]
                e_pers = grp["actual_demand_kw"] - grp["naive_persistence_kw"]
                rmse = float(np.sqrt(np.mean(e ** 2)))
                mae = float(np.mean(np.abs(e)))
                rmse_pers = float(np.sqrt(np.mean(e_pers ** 2)))
                ss_pers = 1.0 - (rmse ** 2) / (rmse_pers ** 2) if rmse_pers > 1e-8 else np.nan
                
                meter_stat_records.append({
                    "district": district,
                    "meter_id": m_id,
                    "exp_id": exp_id,
                    "model": grp["model_architecture"].iloc[0],
                    "feature_config": grp["feature_config"].iloc[0],
                    "rmse_kw": round(rmse, 4),
                    "mae_kw": round(mae, 4),
                    "persistence_rmse_kw": round(rmse_pers, 4),
                    "persistence_skill_score": round(ss_pers, 4),
                    "n_test_intervals": len(grp)
                })

        # Define 15 Pairwise Hypotheses per District Family
        # Family 1: 9 Model-Feature Configurations vs Persistence
        hypotheses = []
        for m in models:
            for f in features:
                exp_id = f"ind_{district}_{m}_{f}"
                hypotheses.append({
                    "name": f"{m}_{f} vs Persistence",
                    "exp1": exp_id,
                    "label1": f"{m}_{f}",
                    "exp2": "naive_persistence",
                    "label2": "Naive Persistence",
                    "category": "Deep Learning vs Persistence"
                })

        # Family 2: Architectural Comparisons (under best feature config: operational weather)
        best_f = "consumption_weather_operational"
        hypotheses.extend([
            {
                "name": f"1D_CNN vs LSTM ({best_f})",
                "exp1": f"ind_{district}_1D_CNN_{best_f}",
                "label1": "1D_CNN (Op Weather)",
                "exp2": f"ind_{district}_LSTM_{best_f}",
                "label2": "LSTM (Op Weather)",
                "category": "Architecture Comparison"
            },
            {
                "name": f"LSTM vs CNN-LSTM ({best_f})",
                "exp1": f"ind_{district}_LSTM_{best_f}",
                "label1": "LSTM (Op Weather)",
                "exp2": f"ind_{district}_CNN_LSTM_{best_f}",
                "label2": "CNN_LSTM (Op Weather)",
                "category": "Architecture Comparison"
            },
            {
                "name": f"1D_CNN vs CNN-LSTM ({best_f})",
                "exp1": f"ind_{district}_1D_CNN_{best_f}",
                "label1": "1D_CNN (Op Weather)",
                "exp2": f"ind_{district}_CNN_LSTM_{best_f}",
                "label2": "CNN_LSTM (Op Weather)",
                "category": "Architecture Comparison"
            }
        ])

        # Family 3: Feature Comparisons (under best architecture)
        # Mathura best: CNN-LSTM; Bareilly best: LSTM
        best_arch = "CNN_LSTM" if district == "mathura" else "LSTM"
        hypotheses.extend([
            {
                "name": f"Consumption vs Retro Weather ({best_arch})",
                "exp1": f"ind_{district}_{best_arch}_consumption_only",
                "label1": f"{best_arch} (Consumption)",
                "exp2": f"ind_{district}_{best_arch}_consumption_weather_retro",
                "label2": f"{best_arch} (Retro Weather)",
                "category": "Feature Set Comparison"
            },
            {
                "name": f"Consumption vs Operational Weather ({best_arch})",
                "exp1": f"ind_{district}_{best_arch}_consumption_only",
                "label1": f"{best_arch} (Consumption)",
                "exp2": f"ind_{district}_{best_arch}_consumption_weather_operational",
                "label2": f"{best_arch} (Operational Weather)",
                "category": "Feature Set Comparison"
            },
            {
                "name": f"Retro vs Operational Weather ({best_arch})",
                "exp1": f"ind_{district}_{best_arch}_consumption_weather_retro",
                "label1": f"{best_arch} (Retro Weather)",
                "exp2": f"ind_{district}_{best_arch}_consumption_weather_operational",
                "label2": f"{best_arch} (Operational Weather)",
                "category": "Feature Set Comparison"
            }
        ])

        # Execute tests for this district
        district_tests = []
        for h in hypotheses:
            diff_series, _ = compute_per_meter_loss_differentials(
                df_dist, h["exp1"], h["exp2"], loss_type="squared_error"
            )
            
            # Primary: Wilcoxon Signed-Rank Test across meters
            wilc = run_meter_wilcoxon_test(diff_series, h["label1"], h["label2"], loss_type="squared_error")
            
            # Supplementary: Meter-Blocked Bootstrap
            boot = run_meter_blocked_bootstrap(diff_series, n_bootstrap=1000)

            pct_model1_beats = float(np.mean(diff_series < 0) * 100.0)

            district_tests.append({
                "district": district,
                "comparison_name": h["name"],
                "category": h["category"],
                "model_1": h["label1"],
                "model_2": h["label2"],
                "n_meters_total": wilc["n_meters_total"],
                "n_meters_evaluated": wilc["n_meters_evaluated"],
                "n_meters_tied": wilc["n_meters_tied"],
                "pct_meters_model1_superior": round(pct_model1_beats, 1),
                "wilcoxon_stat": wilc["statistic"],
                "p_value_raw": wilc["p_value_raw"],
                "median_meter_loss_diff": wilc["median_diff"],
                "mean_meter_loss_diff": wilc["mean_diff"],
                "effect_direction": wilc["effect_direction"],
                "bootstrap_mean": boot["bootstrap_mean"],
                "ci_95_lower": boot["ci_95_lower"],
                "ci_95_upper": boot["ci_95_upper"],
                "bootstrap_p_value": boot["bootstrap_p_value"]
            })

        # Apply Holm-Bonferroni correction within district family
        district_tests_adj = holm_bonferroni_correction(district_tests)
        all_test_records.extend(district_tests_adj)

    df_results = pd.DataFrame(all_test_records)
    df_meter_stats = pd.DataFrame(meter_stat_records)

    out_csv = out_dir / "individual_dm_results.csv"
    out_meter_csv = out_dir / "individual_meter_statistics.csv"
    out_json = out_dir / "individual_dm_summary.json"

    df_results.to_csv(out_csv, index=False)
    df_meter_stats.to_csv(out_meter_csv, index=False)
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json.loads(df_results.to_json(orient="records")), f, indent=2)

    print(f"\nSaved panel DM results to {out_csv}")
    print(f"Saved meter statistics to {out_meter_csv}")
    print("=" * 80)
    return df_results, df_meter_stats

if __name__ == "__main__":
    run_panel_evaluation()

