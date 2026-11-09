"""
Unit Tests for Dataset Slicing and Feature Engineering
Tests extract_contiguous_sequences, temporal_train_val_test_split, and scale_features_and_targets.
"""

import unittest
import numpy as np
import pandas as pd
from src.dataset import (
    extract_contiguous_sequences,
    temporal_train_val_test_split,
    scale_features_and_targets,
    FEATURE_SETS,
)


class TestDatasetSlicing(unittest.TestCase):
    def setUp(self):
        # 100 intervals of sample data with an artificial gap at index 50
        dates = pd.date_range("2020-01-01 00:00", periods=100, freq="15min")
        self.df = pd.DataFrame({
            "timestamp": dates,
            "demand_target": np.linspace(10, 20, 100),
            "retro_temperature_2m": np.linspace(15, 30, 100),
            "retro_relative_humidity_2m": np.linspace(40, 80, 100),
            "retro_dew_point_2m": np.linspace(5, 15, 100),
            "retro_surface_pressure": np.full(100, 985.0),
            "retro_wind_speed_10m": np.full(100, 2.5),
            "retro_direct_radiation": np.linspace(0, 500, 100),
            "retro_diffuse_radiation": np.linspace(0, 200, 100),
            "retro_cloud_cover": np.full(100, 20.0),
            "retro_precipitation": np.zeros(100),
        })
        # Inject NaN at index 50
        self.df.loc[50, "demand_target"] = np.nan

    def test_extract_contiguous_consumption_only(self):
        feature_cols = FEATURE_SETS["consumption_only"]
        X, y, ts = extract_contiguous_sequences(self.df, "demand_target", feature_cols, L=16, H=1)

        self.assertEqual(X.shape[1], 16)
        self.assertEqual(X.shape[2], 1)
        self.assertEqual(y.shape[1], 1)
        # Verify no NaN values in X or y
        self.assertFalse(np.isnan(X).any(), "X must have zero NaNs")
        self.assertFalse(np.isnan(y).any(), "y must have zero NaNs")

    def test_extract_contiguous_weather_retro(self):
        feature_cols = FEATURE_SETS["consumption_weather_retro"]
        X, y, ts = extract_contiguous_sequences(self.df, "demand_target", feature_cols, L=16, H=1)

        self.assertEqual(X.shape[1], 16)
        self.assertEqual(X.shape[2], 10) # demand + 9 weather features
        self.assertFalse(np.isnan(X).any())
        self.assertFalse(np.isnan(y).any())

    def test_temporal_split_order_and_scaling(self):
        feature_cols = FEATURE_SETS["consumption_only"]
        X, y, ts = extract_contiguous_sequences(self.df, "demand_target", feature_cols, L=16, H=1)

        (X_tr, y_tr, ts_tr), (X_va, y_va, ts_va), (X_te, y_te, ts_te) = temporal_train_val_test_split(
            X, y, ts, train_ratio=0.70, val_ratio=0.15
        )

        # Monotonicity check
        self.assertLess(ts_tr[-1], ts_va[0])
        self.assertLess(ts_va[-1], ts_te[0])

        # Scaling check
        (X_tr_s, y_tr_s), (X_va_s, y_va_s), (X_te_s, y_te_s), f_scaler, t_scaler = scale_features_and_targets(
            X_tr, y_tr, X_va, y_va, X_te, y_te
        )

        # Train set mean should be ~0 and std ~1
        self.assertAlmostEqual(float(y_tr_s.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(y_tr_s.std()), 1.0, places=4)

        # Inverse transform should perfectly recover original y
        y_recovered = t_scaler.inverse_transform(y_tr_s)
        np.testing.assert_allclose(y_recovered, y_tr, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

