"""
Unit Tests for Diebold-Mariano Statistical Hypothesis Testing Module
"""

import unittest
import numpy as np
from src.statistical_tests import (
    compute_hac_variance,
    diebold_mariano_test,
    holm_bonferroni_correction
)


class TestStatisticalTests(unittest.TestCase):

    def test_compute_hac_variance_constant_series(self):
        # A constant non-zero loss differential has zero variance around its mean
        d = np.ones(100) * 5.0
        var_d = compute_hac_variance(d, max_lags=4)
        self.assertAlmostEqual(var_d, 0.0, places=6)

    def test_diebold_mariano_identical_predictions(self):
        y_true = np.linspace(10, 50, 100)
        y_pred1 = y_true + 1.0
        y_pred2 = y_true + 1.0 # identical to pred1
        
        res = diebold_mariano_test(y_true, y_pred1, y_pred2, loss_type="squared_error")
        self.assertEqual(res["dm_stat"], 0.0)
        self.assertEqual(res["p_value_raw"], 1.0)
        self.assertEqual(res["effect_direction"], "Identical Losses")

    def test_diebold_mariano_clearly_superior_model(self):
        np.random.seed(42)
        y_true = np.random.normal(50, 10, 200)
        y_pred1 = y_true + np.random.normal(0, 0.1, 200) # near-perfect model
        y_pred2 = y_true + np.random.normal(0, 10.0, 200) # poor model
        
        res = diebold_mariano_test(y_true, y_pred1, y_pred2, loss_type="squared_error")
        # Model 1 has much lower loss -> d_bar < 0 -> DM < 0
        self.assertLess(res["dm_stat"], -5.0)
        self.assertLess(res["p_value_raw"], 1e-5)
        self.assertIn("Model 1 Significantly Superior", res["effect_direction"])

    def test_holm_bonferroni_correction_ordering(self):
        raw_results = [
            {"name": "test_A", "p_value_raw": 0.04},
            {"name": "test_B", "p_value_raw": 0.001},
            {"name": "test_C", "p_value_raw": 0.20}
        ]
        corrected = holm_bonferroni_correction(raw_results)
        
        # Test B had smallest p (0.001), multiplied by 3 -> 0.003
        test_b = [r for r in corrected if r["name"] == "test_B"][0]
        self.assertAlmostEqual(test_b["p_value_adj"], 0.003, places=5)
        self.assertTrue(test_b["significant_fwer_05"])
        
        # Test A had p=0.04, multiplied by 2 -> 0.08 (> 0.05, not significant under FWER)
        test_a = [r for r in corrected if r["name"] == "test_A"][0]
        self.assertAlmostEqual(test_a["p_value_adj"], 0.08, places=5)
        self.assertFalse(test_a["significant_fwer_05"])


if __name__ == "__main__":
    unittest.main()

