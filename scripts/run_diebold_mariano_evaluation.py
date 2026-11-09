"""
Diebold-Mariano Hypothesis Testing Evaluation Runner
Phase 3.2 - Formal Statistical Significance & Pairwise Comparisons
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.statistical_tests import diebold_mariano_test, holm_bonferroni_correction

def run_dm_evaluation():
    pred_path = Path("artifacts/metrics/aggregate_test_predictions.parquet")
    if not pred_path.exists():
        print(f"Error: {pred_path} missing. Run scripts.generate_aggregate_predictions first.")
        return

    df_preds = pd.read_parquet(pred_path)
    print(f"Loaded {len(df_preds)} aggregate test prediction records.")

    districts = ["mathura", "bareilly"]
    records = []

    print("\n" + "=" * 80)
    print("RUNNING PAIRWISE DIEBOLD-MARIANO TESTS (AGGREGATE COHORTS)")
    print("=" * 80)

    for district in districts:
        df_dist = df_preds[df_preds["district"] == district]
        
        # Ground-truth and naive persistence
        # Pick one reference slice to get y_true and y_persistence
        ref_exp = df_dist[df_dist["model_architecture"] == "1D_CNN"]
        ref_exp_sub = ref_exp[ref_exp["feature_config"] == "consumption_only"].sort_values("timestamp")
        
        timestamps = ref_exp_sub["timestamp"].values
        y_true = ref_exp_sub["actual_demand_kw"].values
        y_pers = ref_exp_sub["naive_persistence_kw"].values
        
        # Build lookup for predictions: (model, feature_config) -> y_pred array
        pred_dict = {}
        for (m, f), grp in df_dist.groupby(["model_architecture", "feature_config"]):
            grp_sorted = grp.sort_values("timestamp")
            # Verify timestamp alignment
            assert np.array_equal(grp_sorted["timestamp"].values, timestamps), f"Timestamp mismatch for {m}_{f}"
            pred_dict[(m, f)] = grp_sorted["predicted_demand_kw"].values

        # Find best DL model for this district
        # (Based on lowest test RMSE in consumption-only or overall)
        # For Mathura: LSTM consumption_only (1.5298 kW)
        # For Bareilly: 1D_CNN consumption_only (2.9540 kW)
        if district == "mathura":
            best_dl_key = ("LSTM", "consumption_only")
        else:
            best_dl_key = ("1D_CNN", "consumption_only")

        # Define Prescribed Hypotheses (1 through 7)
        pairwise_hypotheses = [
            # 1. 1D CNN vs LSTM (under consumption_only)
            ("1D_CNN vs LSTM (Consumption Only)", 
             pred_dict[("1D_CNN", "consumption_only")], "1D_CNN",
             pred_dict[("LSTM", "consumption_only")], "LSTM",
             "Model Architecture Comparison"),
             
            # 2. LSTM vs CNN-LSTM (under consumption_only)
            ("LSTM vs CNN-LSTM (Consumption Only)", 
             pred_dict[("LSTM", "consumption_only")], "LSTM",
             pred_dict[("CNN_LSTM", "consumption_only")], "CNN_LSTM",
             "Model Architecture Comparison"),
             
            # 3. 1D CNN vs CNN-LSTM (under consumption_only)
            ("1D_CNN vs CNN-LSTM (Consumption Only)", 
             pred_dict[("1D_CNN", "consumption_only")], "1D_CNN",
             pred_dict[("CNN_LSTM", "consumption_only")], "CNN_LSTM",
             "Model Architecture Comparison"),
             
            # 4. Consumption-only vs Retrospective weather (under best architecture)
            (f"Consumption vs Retro Weather ({best_dl_key[0]})",
             pred_dict[(best_dl_key[0], "consumption_only")], f"{best_dl_key[0]} (Consumption)",
             pred_dict[(best_dl_key[0], "consumption_weather_retro")], f"{best_dl_key[0]} (Retro Weather)",
             "Feature Set Comparison"),
             
            # 5. Consumption-only vs Operational weather (under best architecture)
            (f"Consumption vs Operational Weather ({best_dl_key[0]})",
             pred_dict[(best_dl_key[0], "consumption_only")], f"{best_dl_key[0]} (Consumption)",
             pred_dict[(best_dl_key[0], "consumption_weather_operational")], f"{best_dl_key[0]} (Operational Weather)",
             "Feature Set Comparison"),
             
            # 6. Retrospective vs Operational weather (under best architecture)
            (f"Retro Weather vs Operational Weather ({best_dl_key[0]})",
             pred_dict[(best_dl_key[0], "consumption_weather_retro")], f"{best_dl_key[0]} (Retro Weather)",
             pred_dict[(best_dl_key[0], "consumption_weather_operational")], f"{best_dl_key[0]} (Operational Weather)",
             "Feature Set Comparison"),
             
            # 7. Best DL Model vs Naive Persistence
            (f"Best DL ({best_dl_key[0]}_{best_dl_key[1]}) vs Naive Persistence",
             pred_dict[best_dl_key], f"Best DL ({best_dl_key[0]})",
             y_pers, "Naive Persistence",
             "Deep Learning vs Baseline")
        ]

        # Execute tests for both squared_error (primary) and absolute_error (supplementary)
        cohort_tests = []
        for name, p1, label1, p2, label2, cat in pairwise_hypotheses:
            # Primary: Squared-error
            res_se = diebold_mariano_test(y_true, p1, p2, loss_type="squared_error")
            # Supplementary: Absolute-error
            res_ae = diebold_mariano_test(y_true, p1, p2, loss_type="absolute_error")
            
            cohort_tests.append({
                "district": district,
                "cohort_type": "aggregate",
                "comparison_name": name,
                "category": cat,
                "model_1": label1,
                "model_2": label2,
                "loss_type": "squared_error (primary)",
                "dm_stat": res_se["dm_stat"],
                "p_value_raw": res_se["p_value_raw"],
                "mean_loss_diff": res_se["mean_loss_diff"],
                "loss_model1": res_se["loss_model1"],
                "loss_model2": res_se["loss_model2"],
                "effect_direction": res_se["effect_direction"],
                "sample_size": res_se["sample_size"],
                # Supplementary AE
                "dm_stat_ae": res_ae["dm_stat"],
                "p_value_ae": res_ae["p_value_raw"],
                "effect_ae": res_ae["effect_direction"]
            })

        # Apply Holm-Bonferroni correction within cohort
        cohort_tests_adj = holm_bonferroni_correction(cohort_tests)
        records.extend(cohort_tests_adj)

    # Document Individual Pooled Status (Documented Limitation & Protocol)
    # 17 individual checkpoints reside in Colab; local disk has 1.
    for district in districts:
        records.append({
            "district": district,
            "cohort_type": "individual_pooled",
            "comparison_name": "Individual Pooled DM Suite (7 Hypotheses)",
            "category": "Household Level",
            "model_1": "DL Models",
            "model_2": "Alternative / Persistence",
            "loss_type": "squared_error",
            "dm_stat": np.nan,
            "p_value_raw": np.nan,
            "p_value_adj": np.nan,
            "significant_fwer_05": False,
            "mean_loss_diff": np.nan,
            "loss_model1": np.nan,
            "loss_model2": np.nan,
            "effect_direction": "Pending Colab Checkpoint Export (Documented Limitation)",
            "sample_size": 229281 if district == "mathura" else 398897,
            "dm_stat_ae": np.nan,
            "p_value_ae": np.nan,
            "effect_ae": "Pending Colab Checkpoint Export"
        })

    df_results = pd.DataFrame(records)
    
    # Save CSV and JSON
    out_csv = Path("artifacts/metrics/dm_test_results.csv")
    out_json = Path("artifacts/metrics/dm_test_summary.json")
    df_results.to_csv(out_csv, index=False)
    
    # Clean json export
    summary_dict = {
        "description": "Diebold-Mariano Statistical Significance Tests with Newey-West HAC Variance and Holm-Bonferroni Correction",
        "primary_loss": "Squared-Error Loss (physical kW^2)",
        "supplementary_loss": "Absolute-Error Loss (physical kW)",
        "results": json.loads(df_results.to_json(orient="records"))
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\nSaved DM test results to {out_csv} and {out_json}")

    # Generate Statistical Significance Figure
    plot_dm_results(df_results)

    return df_results


def plot_dm_results(df_results: pd.DataFrame):
    df_agg = df_results[df_results["cohort_type"] == "aggregate"].copy()
    if df_agg.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for idx, district in enumerate(["mathura", "bareilly"]):
        ax = axes[idx]
        sub = df_agg[df_agg["district"] == district].sort_values("dm_stat")
        
        # Color bars by significance and direction
        colors = []
        for _, row in sub.iterrows():
            if row["significant_fwer_05"]:
                colors.append("#2ca02c" if row["dm_stat"] < 0 else "#d62728")
            else:
                colors.append("#7f7f7f") # Insignificant
                
        y_pos = np.arange(len(sub))
        bars = ax.barh(y_pos, sub["dm_stat"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
        
        # Add critical threshold lines (+/- 1.96 for alpha=0.05 asymptotic standard normal)
        ax.axvline(0, color="black", linestyle="-", linewidth=1.0)
        ax.axvline(-1.96, color="blue", linestyle="--", linewidth=1.2, label=r"Critical Threshold ($p=0.05$, $|DM|=1.96$)")
        ax.axvline(1.96, color="blue", linestyle="--", linewidth=1.2)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["comparison_name"], fontsize=9)
        ax.set_xlabel("Diebold-Mariano Statistic (HLN Corrected)", fontsize=11)
        ax.set_title(f"{district.upper()} Aggregate Demand\nPairwise DM Statistics (Primary: Squared Error)", fontsize=12, fontweight="bold")
        
        # Annotate p-values
        for bar, (_, row) in zip(bars, sub.iterrows()):
            width = bar.get_width()
            p_adj_str = f"p_adj={row['p_value_adj']:.3e}" if row['p_value_adj'] < 0.001 else f"p_adj={row['p_value_adj']:.4f}"
            offset = 0.5 if width >= 0 else -0.5
            ha = "left" if width >= 0 else "right"
            ax.text(width + offset, bar.get_y() + bar.get_height() / 2, p_adj_str,
                    va="center", ha=ha, fontsize=8, fontweight="bold" if row["significant_fwer_05"] else "normal")

        if idx == 0:
            ax.legend(loc="lower right", frameon=True)

    plt.suptitle("Diebold-Mariano Predictive Accuracy Tests with Newey-West HAC Variance & Holm Correction",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig_path = Path("artifacts/metrics/dm_significance_comparison.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved statistical significance plot to {fig_path}")

if __name__ == "__main__":
    run_dm_evaluation()

