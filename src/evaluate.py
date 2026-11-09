"""
Evaluation Metrics Module for Short-Term Electricity Demand Forecasting
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models
"""

import numpy as np
from typing import Dict, Optional


def compute_forecasting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persistence: Optional[np.ndarray] = None,
    peak_quantile: float = 0.95
) -> Dict[str, Optional[float]]:
    """
    Computes rigorous statistical regression and power forecasting benchmark metrics.
    All error metrics (RMSE, MAE, Peak MAE) are expressed in physical kW units.
    
    Args:
        y_true: Array of actual observed demand values in physical kW.
        y_pred: Array of model predicted demand values in physical kW.
        y_persistence: Optional array of naive persistence predictions for skill score computation.
        peak_quantile: Top load quantile considered as peak hours (default: 0.95).
        
    Returns:
        Dict with RMSE, MAE, WAPE, MAPE, R2, nRMSE, Persistence_Skill_Score, and Peak_MAE.
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    
    # Assert identical non-empty shapes
    assert len(y_t) == len(y_p), f"Shape mismatch: y_true {y_t.shape} vs y_pred {y_p.shape}"
    assert len(y_t) > 0, "Empty evaluation arrays passed"

    # 1. Root Mean Squared Error (RMSE) in physical kW
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    
    # 2. Mean Absolute Error (MAE) in physical kW
    mae = float(np.mean(np.abs(y_t - y_p)))
    
    # 3. Weighted Absolute Percentage Error (WAPE) in %
    # WAPE = sum(|y - y_hat|) / sum(y) * 100.
    # Operates across all intervals including zero-load without asymptotic division-by-zero.
    sum_yt = np.sum(y_t)
    if sum_yt > 1e-6:
        wape = float((np.sum(np.abs(y_t - y_p)) / sum_yt) * 100.0)
    else:
        wape = float("nan")

    # 4. Mean Absolute Percentage Error (MAPE) in %
    # Excludes near-zero observations (y < 0.01 kW) to prevent asymptotic distortion during zero-load intervals.
    non_zero_mask = y_t >= 0.01
    if non_zero_mask.sum() > 0:
        mape = float(np.mean(np.abs((y_t[non_zero_mask] - y_p[non_zero_mask]) / y_t[non_zero_mask])) * 100.0)
    else:
        mape = float("nan")
        
    # 5. Coefficient of Determination (R^2)
    # Measures the proportion of variance explained by the model relative to the sample mean.
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    
    # 6. Normalized RMSE (nRMSE) in %
    # Normalized by the empirical mean of actual observed demand.
    mean_val = float(np.mean(y_t))
    nrmse = float((rmse / (mean_val + 1e-8)) * 100.0)
    
    # 7. Persistence Skill Score (SS_pers)
    # SS_pers = 1 - (RMSE_model^2 / RMSE_persistence^2)
    # Measures improvement strictly relative to the persistence baseline.
    persistence_skill_score = None
    if y_persistence is not None:
        y_pers = np.asarray(y_persistence, dtype=np.float64).flatten()
        assert len(y_pers) == len(y_t), "Persistence array length mismatch"
        rmse_pers = float(np.sqrt(np.mean((y_t - y_pers) ** 2)))
        if rmse_pers > 1e-8:
            persistence_skill_score = float(1.0 - ((rmse ** 2) / (rmse_pers ** 2)))

    # 8. Peak Load Evaluation (Top 5% Highest Actual Demand Intervals)
    peak_threshold = np.quantile(y_t, peak_quantile)
    peak_mask = y_t >= peak_threshold
    if peak_mask.sum() > 0:
        peak_mae = float(np.mean(np.abs(y_t[peak_mask] - y_p[peak_mask])))
        peak_mape = float(np.mean(np.abs((y_t[peak_mask] - y_p[peak_mask]) / y_t[peak_mask])) * 100.0)
    else:
        peak_mae = float("nan")
        peak_mape = float("nan")
        
    return {
        "rmse_kw": round(rmse, 4),
        "mae_kw": round(mae, 4),
        "wape_pct": round(wape, 2) if not np.isnan(wape) else None,
        "mape_pct": round(mape, 2) if not np.isnan(mape) else None,
        "r2_score": round(r2, 4),
        "nrmse_pct": round(nrmse, 2),
        "persistence_skill_score": round(persistence_skill_score, 4) if persistence_skill_score is not None else None,
        "peak_mae_kw": round(peak_mae, 4) if not np.isnan(peak_mae) else None,
        "peak_mape_pct": round(peak_mape, 2) if not np.isnan(peak_mape) else None,
        "mape_exclusion_threshold_kw": 0.01,
    }
