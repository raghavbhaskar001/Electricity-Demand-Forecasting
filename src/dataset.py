"""
Dataset Preparation & Sequence Slicing Module for Deep Learning Time-Series Models
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


FEATURE_SETS = {
    "consumption_only": [
        "demand_target"
    ],
    "consumption_weather_retro": [
        "demand_target",
        "retro_temperature_2m",
        "retro_relative_humidity_2m",
        "retro_dew_point_2m",
        "retro_surface_pressure",
        "retro_wind_speed_10m",
        "retro_direct_radiation",
        "retro_diffuse_radiation",
        "retro_cloud_cover",
        "retro_precipitation",
    ],
    "consumption_weather_operational": [
        "demand_target",
        "operational_temperature_2m",
        "operational_relative_humidity_2m",
        "operational_dew_point_2m",
        "operational_surface_pressure",
        "operational_wind_speed_10m",
        "operational_direct_radiation",
        "operational_diffuse_radiation",
        "operational_cloud_cover",
        "operational_precipitation",
    ],
}

CALENDAR_FEATURES = [
    "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"
]


def extract_contiguous_sequences(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    lookback_len: int = 16,
    horizon_len: int = 1,
    L: Optional[int] = None,
    H: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts strictly contiguous sequence pairs (X, y) that do NOT cross missing/invalid intervals.
    
    Args:
        df: DataFrame sorted chronologically along master calendar grid.
        target_col: Column name of target demand (kW).
        feature_cols: List of feature column names.
        lookback_len: Number of past time steps (L=16 = 4 hours).
        horizon_len: Single-step forecast horizon (H=1 = 15 min).
        L: Optional alias for lookback_len.
        H: Optional alias for horizon_len.

    Returns:
        X: Array of shape (num_samples, lookback_len, num_features)
        y: Array of shape (num_samples, 1)
        timestamps: Array of target timestamps for evaluation tracking.
    """
    if L is not None:
        lookback_len = L
    if H is not None:
        horizon_len = H
    # Ensure target_col is mapped in feature_cols if needed
    sub_df = df.copy()
    if "demand_target" in feature_cols and "demand_target" not in sub_df.columns:
        sub_df["demand_target"] = sub_df[target_col]

    all_cols = list(dict.fromkeys(feature_cols + [target_col]))
    
    # Check completeness mask across all required features + target
    valid_mask = sub_df[all_cols].notna().all(axis=1).values
    total_len = len(sub_df)
    window_size = lookback_len + horizon_len

    if total_len < window_size:
        return np.empty((0, lookback_len, len(feature_cols)), dtype=np.float32), np.empty((0, 1), dtype=np.float32), np.empty((0,))

    # Vectorized sliding window validation check
    from numpy.lib.stride_tricks import sliding_window_view
    valid_windows = sliding_window_view(valid_mask, window_size).all(axis=-1)
    valid_indices = np.where(valid_windows)[0]

    if len(valid_indices) == 0:
        return np.empty((0, lookback_len, len(feature_cols)), dtype=np.float32), np.empty((0, 1), dtype=np.float32), np.empty((0,))

    feat_values = sub_df[feature_cols].values
    target_values = sub_df[target_col].values
    time_values = sub_df["timestamp"].values

    # Vectorized sliding window tensor extraction
    feat_windows = sliding_window_view(feat_values, window_shape=(lookback_len, len(feature_cols)))[:, 0, :, :]
    X = np.copy(feat_windows[valid_indices]).astype(np.float32)
    y = np.copy(target_values[valid_indices + lookback_len + horizon_len - 1]).astype(np.float32).reshape(-1, 1)
    timestamps = np.copy(time_values[valid_indices + lookback_len + horizon_len - 1])
    return X, y, timestamps


def temporal_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge_gap: int = 0
):
    """
    Performs chronological non-shuffled train / validation / test split.
    Prevents temporal data leakage.

    Args:
        X: Sequence input array of shape (num_samples, seq_len, num_features).
        y: Target array of shape (num_samples, 1).
        timestamps: Target timestamps array of shape (num_samples,).
        train_ratio: Fraction for training partition (default: 0.70).
        val_ratio: Fraction for validation partition (default: 0.15).
        purge_gap: Number of sequence steps to purge immediately after the train
                   and validation boundaries. When purge_gap = L (e.g. 16),
                   no sequence windows in validation overlap with training observations,
                   and no sequence windows in test overlap with validation observations.
    """
    n = len(X)
    n_train = int(n * train_ratio)
    val_start = min(n, n_train + purge_gap)
    n_val = int(n * val_ratio)
    val_end = min(n, val_start + n_val)
    test_start = min(n, val_end + purge_gap)

    X_train, y_train, ts_train = X[:n_train], y[:n_train], timestamps[:n_train]
    X_val, y_val, ts_val = X[val_start:val_end], y[val_start:val_end], timestamps[val_start:val_end]
    X_test, y_test, ts_test = X[test_start:], y[test_start:], timestamps[test_start:]

    return (X_train, y_train, ts_train), (X_val, y_val, ts_val), (X_test, y_test, ts_test)


def scale_features_and_targets(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray
):
    """
    Fits standard scalers strictly on training data only.
    Transforms validation and test sets without leakage.
    """
    n_samples, seq_len, n_features = X_train.shape
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Reshape X to 2D for scaler fitting
    X_train_2d = X_train.reshape(-1, n_features)
    feature_scaler.fit(X_train_2d)

    target_scaler.fit(y_train)

    # Transform X
    X_train_scaled = feature_scaler.transform(X_train_2d).reshape(n_samples, seq_len, n_features)
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, n_features)).reshape(len(X_val), seq_len, n_features)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, n_features)).reshape(len(X_test), seq_len, n_features)

    # Transform y
    y_train_scaled = target_scaler.transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)

    return (
        (X_train_scaled, y_train_scaled),
        (X_val_scaled, y_val_scaled),
        (X_test_scaled, y_test_scaled),
        feature_scaler,
        target_scaler
    )


if TORCH_AVAILABLE:
    class ElectricityTimeSeriesDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
