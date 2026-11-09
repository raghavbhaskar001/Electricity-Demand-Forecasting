"""
Unit Tests for Baselines, Metrics, and Boundary Purge Slicing
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models
"""

import unittest
import numpy as np
import pandas as pd
from src.dataset import (
    extract_contiguous_sequences,
    temporal_train_val_test_split,
)
from src.evaluate import compute_forecasting_metrics
from src.baselines import NaivePersistence, SeasonalPersistence


class TestBaselinesAndMetrics(unittest.TestCase):
    def setUp(self):
        # Synthetic sine wave demand series
        self.dates = pd.date_range("2020-01-01 00:00", periods=200, freq="15min")
        self.t = np.arange(200)
        # Period 96 (daily seasonality) + base load
        self.demand = 5.0 + 3.0 * np.sin(2 * np.pi * self.t / 96) + 0.1 * self.t / 200.0
        self.df = pd.DataFrame({
            "timestamp": self.dates,
            "demand_target": self.demand
        })

    def test_persistence_skill_score_identical_persistence(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pers = np.array([0.9, 1.9, 3.1, 3.9])
        # If model is identical to persistence, skill score must be 0.0
        metrics = compute_forecasting_metrics(y_true, y_pers, y_persistence=y_pers)
        self.assertAlmostEqual(metrics["persistence_skill_score"], 0.0, places=4)

    def test_persistence_skill_score_perfect_model(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0]) # Perfect
        y_pers = np.array([0.5, 1.5, 2.5, 3.5]) # Imperfect
        metrics = compute_forecasting_metrics(y_true, y_pred, y_persistence=y_pers)
        self.assertAlmostEqual(metrics["persistence_skill_score"], 1.0, places=4)

    def test_wape_and_zero_load_guard(self):
        y_true = np.array([0.0, 0.005, 1.0, 2.0])
        y_pred = np.array([0.1, 0.002, 1.1, 1.9])
        # MAPE should ignore the two near-zero points (<0.01 kW) without throwing error or NaN
        metrics = compute_forecasting_metrics(y_true, y_pred, y_persistence=y_pred)
        self.assertIsNotNone(metrics["wape_pct"])
        self.assertIsNotNone(metrics["mape_pct"])
        self.assertGreater(metrics["wape_pct"], 0.0)

    def test_boundary_purge_strictly_eliminates_overlap(self):
        L = 16
        H = 1
        X, y, ts = extract_contiguous_sequences(
            self.df, target_col="demand_target", feature_cols=["demand_target"],
            lookback_len=L, horizon_len=H
        )

        (X_tr, y_tr, ts_tr), (X_va, y_va, ts_va), (X_te, y_te, ts_te) = temporal_train_val_test_split(
            X, y, ts, train_ratio=0.70, val_ratio=0.15, purge_gap=L
        )

        # In time series with 15-min intervals:
        # Last target timestamp in train
        last_train_target_ts = pd.to_datetime(ts_tr[-1])
        # First target timestamp in val
        first_val_target_ts = pd.to_datetime(ts_va[0])
        
        # When purge_gap = L (16), the distance between last train target and first val target
        # must be strictly > L * 15 minutes!
        time_delta_min = (first_val_target_ts - last_train_target_ts).total_seconds() / 60.0
        expected_min_gap = (L + 1) * 15.0  # At least 17 intervals = 255 minutes
        self.assertGreaterEqual(time_delta_min, expected_min_gap)

    def test_naive_persistence_logic(self):
        X = np.arange(32).reshape(2, 16, 1) # 2 samples, L=16, F=1
        naive = NaivePersistence()
        pred = naive.predict_from_sequences(X)
        self.assertEqual(pred.shape, (2, 1))
        # The prediction should equal the last step in each sequence
        self.assertEqual(pred[0, 0], X[0, -1, 0])
        self.assertEqual(pred[1, 0], X[1, -1, 0])


if __name__ == "__main__":
    unittest.main()

