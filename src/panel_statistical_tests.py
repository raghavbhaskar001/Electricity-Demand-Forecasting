"""
Panel Statistical Evaluation Module for Pooled Smart Meter Panels
Phase 3.3 - Meter-Level Paired Comparisons & Cluster-Robust Inference

Implements:
1. Per-meter paired loss differential aggregation.
2. Wilcoxon signed-rank test across household smart meters.
3. Meter-blocked cluster bootstrap accounting for intra-household temporal dependence.
4. Holm-Bonferroni step-down correction for panel hypothesis families.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any


def compute_per_meter_loss_differentials(
    df_preds: pd.DataFrame,
    exp_id_1: str,
    exp_id_2: str,
    loss_type: str = "squared_error"
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Computes per-meter aggregated losses and paired loss differentials D_m = L1_m - L2_m.
    
    Args:
        df_preds: DataFrame containing test predictions with columns:
                  [timestamp, district, meter_id, actual_demand_kw, predicted_demand_kw, exp_id]
        exp_id_1: Experiment ID or 'naive_persistence' for Model 1.
        exp_id_2: Experiment ID or 'naive_persistence' for Model 2.
        loss_type: 'squared_error' (default) or 'absolute_error'.
        
    Returns:
        (diff_series, meter_summary_df)
    """
    # Filter Model 1
    if exp_id_1 == "naive_persistence":
        sub1 = df_preds[df_preds["exp_id"] == df_preds["exp_id"].iloc[0]].copy()
        sub1["pred"] = sub1["naive_persistence_kw"]
    else:
        sub1 = df_preds[df_preds["exp_id"] == exp_id_1].copy()
        sub1["pred"] = sub1["predicted_demand_kw"]

    # Filter Model 2
    if exp_id_2 == "naive_persistence":
        sub2 = df_preds[df_preds["exp_id"] == df_preds["exp_id"].iloc[0]].copy()
        sub2["pred"] = sub2["naive_persistence_kw"]
    else:
        sub2 = df_preds[df_preds["exp_id"] == exp_id_2].copy()
        sub2["pred"] = sub2["predicted_demand_kw"]

    # Compute errors
    if loss_type == "squared_error":
        sub1["loss"] = (sub1["actual_demand_kw"] - sub1["pred"]) ** 2
        sub2["loss"] = (sub2["actual_demand_kw"] - sub2["pred"]) ** 2
    elif loss_type == "absolute_error":
        sub1["loss"] = np.abs(sub1["actual_demand_kw"] - sub1["pred"])
        sub2["loss"] = np.abs(sub2["actual_demand_kw"] - sub2["pred"])
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Aggregate mean loss per meter
    m_loss1 = sub1.groupby("meter_id")["loss"].mean()
    m_loss2 = sub2.groupby("meter_id")["loss"].mean()

    # Align on common meters
    common_meters = m_loss1.index.intersection(m_loss2.index)
    loss1_aligned = m_loss1.loc[common_meters]
    loss2_aligned = m_loss2.loc[common_meters]

    diff_series = loss1_aligned - loss2_aligned

    meter_summary = pd.DataFrame({
        "meter_id": common_meters,
        "loss_model1": loss1_aligned.values,
        "loss_model2": loss2_aligned.values,
        "loss_differential": diff_series.values
    }).set_index("meter_id")

    return diff_series, meter_summary


def run_meter_wilcoxon_test(
    diff_series: pd.Series,
    label1: str,
    label2: str,
    loss_type: str = "squared_error"
) -> Dict[str, Any]:
    """
    Performs Wilcoxon signed-rank test across meters on paired loss differentials.
    H0: Median(D_m) = 0.
    """
    # Drop zero differences
    diffs = diff_series.dropna().values
    n_total = len(diffs)
    non_zero_diffs = diffs[diffs != 0.0]
    n_used = len(non_zero_diffs)
    n_zero = n_total - n_used

    if n_used < 5:
        return {
            "test_type": "Wilcoxon Signed-Rank (Primary)",
            "n_meters_total": n_total,
            "n_meters_evaluated": n_used,
            "n_meters_tied": n_zero,
            "statistic": float("nan"),
            "p_value_raw": float("nan"),
            "median_diff": float("nan"),
            "mean_diff": float("nan"),
            "effect_direction": "Insufficient Non-Zero Differences",
            "loss_type": loss_type
        }

    stat, p_val = stats.wilcoxon(non_zero_diffs, alternative="two-sided")
    med_diff = float(np.median(diffs))
    mean_diff = float(np.mean(diffs))

    # Effect direction
    if p_val < 0.05:
        if med_diff < 0:
            effect = f"{label1} Statistically Superior (Lower Median Loss)"
        else:
            effect = f"{label2} Statistically Superior (Lower Median Loss)"
    else:
        effect = "No Statistically Significant Difference (p >= 0.05)"

    return {
        "test_type": "Wilcoxon Signed-Rank (Primary)",
        "n_meters_total": n_total,
        "n_meters_evaluated": n_used,
        "n_meters_tied": n_zero,
        "statistic": round(float(stat), 4),
        "p_value_raw": float(p_val),
        "median_diff": round(med_diff, 6),
        "mean_diff": round(mean_diff, 6),
        "effect_direction": effect,
        "loss_type": loss_type
    }


def run_meter_blocked_bootstrap(
    meter_diffs: pd.Series,
    n_bootstrap: int = 1000,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Performs meter-blocked cluster bootstrap.
    Resamples entire meters with replacement to account for intra-meter serial autocorrelation.
    """
    rng = np.random.default_rng(random_seed)
    diffs = meter_diffs.dropna().values
    M = len(diffs)
    
    if M < 5:
        return {
            "test_type": "Meter-Blocked Bootstrap (Supplementary)",
            "bootstrap_mean": float("nan"),
            "ci_95_lower": float("nan"),
            "ci_95_upper": float("nan"),
            "bootstrap_p_value": float("nan")
        }

    boot_means = []
    for _ in range(n_bootstrap):
        sampled_diffs = rng.choice(diffs, size=M, replace=True)
        boot_means.append(np.mean(sampled_diffs))
        
    boot_means = np.array(boot_means)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    
    # Two-sided empirical bootstrap p-value
    # Proportion of resamples on opposite side of 0 from sample mean
    obs_mean = np.mean(diffs)
    if obs_mean > 0:
        p_boot = float(2.0 * np.mean(boot_means <= 0))
    else:
        p_boot = float(2.0 * np.mean(boot_means >= 0))
    p_boot = min(1.0, max(0.0, p_boot))

    return {
        "test_type": "Meter-Blocked Bootstrap (Supplementary)",
        "bootstrap_mean": round(float(np.mean(boot_means)), 6),
        "ci_95_lower": round(ci_lower, 6),
        "ci_95_upper": round(ci_upper, 6),
        "bootstrap_p_value": round(p_boot, 4)
    }

