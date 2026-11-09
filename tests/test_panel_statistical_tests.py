"""
Unit Tests for Panel Statistical Tests Module
"""

import unittest
import numpy as np
import pandas as pd
from src.panel_statistical_tests import (
    compute_per_meter_loss_differentials,
    run_meter_wilcoxon_test,
    run_meter_blocked_bootstrap
)

class TestPanelStatisticalTests(unittest.TestCase):

    def setUp(self):
        # Create synthetic test predictions
        np.random.seed(42)
        records = []
        meters = [f"meter_{i:02d}" for i in range(20)]
        for m in meters:
            y_true = np.random.uniform(0.1, 2.0, 50)
            pred1 = y_true + np.random.normal(0, 0.05, 50) # superior model
            pred2 = y_true + np.random.normal(0, 0.20, 50) # inferior model
            naive = y_true + np.random.normal(0, 0.15, 50)
            
            for t in range(50):
                records.append({
                    "timestamp": pd.Timestamp("2020-03-01") + pd.Timedelta(minutes=15 * t),
                    "district": "mathura",
                    "meter_id": m,
                    "actual_demand_kw": y_true[t],
                    "predicted_demand_kw": pred1[t],
                    "naive_persistence_kw": naive[t],
                    "exp_id": "model_1"
                })
                records.append({
                    "timestamp": pd.Timestamp("2020-03-01") + pd.Timedelta(minutes=15 * t),
                    "district": "mathura",
                    "meter_id": m,
                    "actual_demand_kw": y_true[t],
                    "predicted_demand_kw": pred2[t],
                    "naive_persistence_kw": naive[t],
                    "exp_id": "model_2"
                })
        self.df_preds = pd.DataFrame(records)

    def test_compute_differentials(self):
        diffs, summary = compute_per_meter_loss_differentials(
            self.df_preds, "model_1", "model_2", loss_type="squared_error"
        )
        self.assertEqual(len(diffs), 20)
        # Model 1 has much lower noise -> all diffs < 0
        self.assertTrue((diffs < 0).all())

    def test_wilcoxon_test_superiority(self):
        diffs, _ = compute_per_meter_loss_differentials(
            self.df_preds, "model_1", "model_2", loss_type="squared_error"
        )
        res = run_meter_wilcoxon_test(diffs, "Model 1", "Model 2")
        self.assertEqual(res["n_meters_total"], 20)
        self.assertEqual(res["n_meters_evaluated"], 20)
        self.assertLess(res["p_value_raw"], 1e-4)
        self.assertIn("Model 1 Statistically Superior", res["effect_direction"])

    def test_meter_blocked_bootstrap(self):
        diffs, _ = compute_per_meter_loss_differentials(
            self.df_preds, "model_1", "model_2", loss_type="squared_error"
        )
        boot = run_meter_blocked_bootstrap(diffs, n_bootstrap=200, random_seed=42)
        self.assertLess(boot["ci_95_upper"], 0.0) # upper CI strictly negative
        self.assertLess(boot["bootstrap_p_value"], 0.01)

if __name__ == "__main__":
    unittest.main()

