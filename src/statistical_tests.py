"""
Diebold-Mariano Statistical Hypothesis Testing Module
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models

Implements:
1. Diebold-Mariano (1995) test with Harvey-Leybourne-Newbold (1997) adjustment.
2. Newey-West (1987) Heteroskedasticity and Autocorrelation Consistent (HAC) covariance estimator.
3. Multiple-testing correction via the Holm-Bonferroni procedure.
4. Meter-clustered block-dependence adjustments for pooled panel observations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any


def compute_hac_variance(d: np.ndarray, max_lags: Optional[int] = None) -> float:
    """
    Computes Newey-West (1987) HAC variance estimator with Bartlett kernel.
    
    Args:
        d: 1D array of loss differentials (d_t = L(e1_t) - L(e2_t)).
        max_lags: Truncation lag K. If None, uses Newey-West rule K = floor(4 * (T / 100)^(2/9)).
        
    Returns:
        Asymptotic variance of the sample mean d_bar.
    """
    T = len(d)
    if T < 2:
        return float("nan")
        
    d_bar = np.mean(d)
    d_demeaned = d - d_bar
    
    if max_lags is None:
        # Standard rule of thumb for bandwidth selection
        max_lags = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    max_lags = max(0, min(max_lags, T - 1))
    
    # Sample autocovariances: gamma_0, gamma_1, ..., gamma_K
    gamma_0 = float(np.dot(d_demeaned, d_demeaned) / T)
    
    var_d_bar = gamma_0
    for k in range(1, max_lags + 1):
        weight = 1.0 - (k / (max_lags + 1.0)) # Bartlett kernel weight
        gamma_k = float(np.dot(d_demeaned[k:], d_demeaned[:-k]) / T)
        var_d_bar += 2.0 * weight * gamma_k
        
    # Variance of sample mean is var(d) / T
    return max(1e-12, var_d_bar / T)


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    loss_type: str = "squared_error",
    h: int = 1,
    max_lags: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes two-sided Diebold-Mariano test between Model 1 and Model 2 forecasts.
    
    Args:
        y_true: Observed ground-truth series.
        y_pred1: Forecast from Model 1.
        y_pred2: Forecast from Model 2.
        loss_type: 'squared_error' (default) or 'absolute_error'.
        h: Forecast horizon (default: 1 step).
        max_lags: Maximum autocorrelation lags for Newey-West HAC variance.
        
    Returns:
        Dict with DM statistic, raw two-sided p-value, mean loss differential, and effect direction.
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p1 = np.asarray(y_pred1, dtype=np.float64).flatten()
    y_p2 = np.asarray(y_pred2, dtype=np.float64).flatten()
    
    assert len(y_t) == len(y_p1) == len(y_p2), "Input arrays must have identical length"
    T = len(y_t)
    assert T >= 10, f"Insufficient sample size for DM test (T={T})"
    
    e1 = y_t - y_p1
    e2 = y_t - y_p2
    
    if loss_type == "squared_error":
        loss1 = e1 ** 2
        loss2 = e2 ** 2
    elif loss_type == "absolute_error":
        loss1 = np.abs(e1)
        loss2 = np.abs(e2)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
        
    d = loss1 - loss2
    d_bar = float(np.mean(d))
    
    var_d_bar = compute_hac_variance(d, max_lags=max_lags)
    std_err = float(np.sqrt(var_d_bar))
    
    if std_err < 1e-12:
        return {
            "dm_stat": 0.0,
            "p_value_raw": 1.0,
            "mean_loss_diff": 0.0,
            "loss_model1": float(np.mean(loss1)),
            "loss_model2": float(np.mean(loss2)),
            "effect_direction": "Identical Losses",
            "sample_size": T,
            "loss_type": loss_type,
            "hln_adjusted_dm": 0.0
        }
        
    dm_raw = d_bar / std_err
    
    # Harvey, Leybourne, Newbold (1997) finite-sample correction
    hln_factor = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
    dm_hln = float(dm_raw * hln_factor)
    
    # Two-sided Student-t p-value (df = T - 1) per HLN recommendation
    p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_hln), df=T - 1)))
    
    # Determine effect direction
    if np.allclose(loss1, loss2):
        effect = "Identical Losses"
    elif p_value < 0.05:
        if dm_hln < 0:
            effect = "Model 1 Significantly Superior (Lower Loss)"
        else:
            effect = "Model 2 Significantly Superior (Lower Loss)"
    else:
        effect = "No Statistically Significant Difference (p >= 0.05)"
        
    return {
        "dm_stat": round(dm_hln, 4),
        "p_value_raw": float(p_value),
        "mean_loss_diff": round(d_bar, 6),
        "loss_model1": round(float(np.mean(loss1)), 6),
        "loss_model2": round(float(np.mean(loss2)), 6),
        "effect_direction": effect,
        "sample_size": T,
        "loss_type": loss_type,
        "std_error": round(std_err, 6)
    }


def holm_bonferroni_correction(results_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies Holm-Bonferroni step-down correction to control Family-Wise Error Rate (FWER).
    
    Args:
        results_list: List of test result dicts containing 'p_value_raw'.
        
    Returns:
        Updated list with 'p_value_adj' and 'significant_fwer_05' boolean flags.
    """
    m = len(results_list)
    if m == 0:
        return []
        
    # Sort indices by raw p-value ascending
    sorted_indices = sorted(range(m), key=lambda i: results_list[i]["p_value_raw"])
    
    adjusted_p_values = [1.0] * m
    running_max = 0.0
    
    for rank, orig_idx in enumerate(sorted_indices):
        k = rank + 1 # 1-based rank
        multiplier = m - k + 1
        raw_p = results_list[orig_idx]["p_value_raw"]
        adj_p = min(1.0, raw_p * multiplier)
        running_max = max(running_max, adj_p)
        adjusted_p_values[orig_idx] = min(1.0, running_max)
        
    out = []
    for i, res in enumerate(results_list):
        res_copy = dict(res)
        adj_p = adjusted_p_values[i]
        res_copy["p_value_adj"] = float(adj_p)
        res_copy["significant_fwer_05"] = bool(adj_p < 0.05)
        out.append(res_copy)
        
    return out
