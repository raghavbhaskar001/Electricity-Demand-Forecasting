"""
Comprehensive unit and reconciliation tests for the Interactive Electricity
Demand Prediction Simulator (`src/simulator.py`).
"""

import unittest
import numpy as np
import pandas as pd

from src.dataset import FEATURE_SETS
from src.simulator import (
    LOOKBACK_STEPS,
    SCENARIO_PRESETS,
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_DISTRICTS,
    SUPPORTED_FEATURE_CONFIGS,
    PredictionSimulator,
    SimulatorValidationError,
    UnsupportedConfigurationError,
)


class TestPredictionSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = PredictionSimulator()
        cls.saved_preds = pd.read_parquet("artifacts/metrics/aggregate_test_predictions.parquet")

    def test_availability_matrix_completeness_and_integrity(self) -> None:
        matrix = self.sim.get_availability_matrix()
        self.assertEqual(len(matrix), 36)

        supported = [r for r in matrix if r["supported"]]
        unsupported = [r for r in matrix if not r["supported"]]
        self.assertEqual(len(supported), 18)
        self.assertEqual(len(unsupported), 18)

        # Every aggregate model is supported; every individual model is unsupported
        self.assertTrue(all(r["scale"] == "aggregate" for r in supported))
        self.assertTrue(all(r["scale"] == "individual" for r in unsupported))

        # Interrupted epoch-3 local individual checkpoint MUST be explicitly excluded
        partial = next(r for r in matrix if r["exp_id"] == "ind_mathura_1D_CNN_consumption_only")
        self.assertFalse(partial["supported"])
        self.assertEqual(partial["status"], "partial_checkpoint_excluded")

    def test_unsupported_combinations_blocked(self) -> None:
        valid_16 = [15.0] * LOOKBACK_STEPS
        with self.assertRaises(UnsupportedConfigurationError):
            self.sim.predict_next_interval(
                "individual", "mathura", "1D_CNN", "consumption_only", valid_16
            )
        with self.assertRaises(UnsupportedConfigurationError):
            self.sim.predict_next_interval(
                "individual", "bareilly", "LSTM", "consumption_only", valid_16
            )
        with self.assertRaises(UnsupportedConfigurationError):
            self.sim.predict_next_interval(
                "aggregate", "delhi", "LSTM", "consumption_only", valid_16
            )

    def test_exact_reconciliation_all_18_aggregate_models(self) -> None:
        """Verify simulator predictions match saved test-set predictions across all 18 models."""
        for district in SUPPORTED_DISTRICTS:
            ctx = self.sim._get_context(district)
            for f_cfg in SUPPORTED_FEATURE_CONFIGS:
                X_te = ctx.X_test_by_config[f_cfg]
                for arch in SUPPORTED_ARCHITECTURES:
                    exp_id = f"agg_{district}_{arch}_{f_cfg}"
                    sub = self.saved_preds[self.saved_preds["exp_id"] == exp_id].sort_values("timestamp")
                    for idx in (0, len(X_te) // 2, len(X_te) - 1):
                        cons = X_te[idx, :, 0]
                        wx = X_te[idx, :, 1:] if X_te.shape[2] > 1 else None
                        res = self.sim.predict_next_interval(
                            "aggregate", district, arch, f_cfg, cons, wx
                        )
                        expected_kw = float(sub["predicted_demand_kw"].iloc[idx])
                        self.assertAlmostEqual(
                            res["predicted_demand_kw"],
                            expected_kw,
                            places=4,
                            msg=f"Mismatch in {exp_id} at test index {idx}",
                        )

    def test_manual_input_validation_rejects_invalid_sequences(self) -> None:
        # Wrong lengths
        with self.assertRaises(SimulatorValidationError):
            self.sim.validate_inputs("mathura", "consumption_only", [18.0] * 15)
        with self.assertRaises(SimulatorValidationError):
            self.sim.validate_inputs("mathura", "consumption_only", [18.0] * 17)

        # Missing / non-numeric / NaN / Inf / negative
        bad_payloads = [
            [18.0] * 15 + [None],
            [18.0] * 15 + [""],
            [18.0] * 15 + ["abc"],
            [18.0] * 15 + [float("nan")],
            [18.0] * 15 + [float("inf")],
            [18.0] * 15 + [-0.5],
        ]
        for payload in bad_payloads:
            with self.assertRaises(SimulatorValidationError):
                self.sim.validate_inputs("mathura", "consumption_only", payload)

    def test_weather_configurations_never_fabricate_weather(self) -> None:
        cons = [18.0] * LOOKBACK_STEPS
        for wx_cfg in ("consumption_weather_retro", "consumption_weather_operational"):
            with self.assertRaises(SimulatorValidationError):
                self.sim.predict_next_interval(
                    "aggregate", "mathura", "LSTM", wx_cfg, cons, weather_rows=None
                )
            # Wrong column count (e.g. 5 instead of 9)
            with self.assertRaises(SimulatorValidationError):
                self.sim.predict_next_interval(
                    "aggregate", "mathura", "LSTM", wx_cfg, cons, weather_rows=[[25.0] * 5] * 16
                )

    def test_ood_warning_without_silent_clipping(self) -> None:
        extreme = [18.0] * 15 + [92.5]  # Above Mathura training max (39.06 kW)
        res = self.sim.predict_next_interval(
            "aggregate", "mathura", "1D_CNN", "consumption_only", extreme
        )
        val = res["validation"]
        self.assertTrue(val["valid"])
        self.assertFalse(val["clipped"])
        self.assertEqual(val["status"], "valid_with_ood_warning")
        self.assertAlmostEqual(res["historical_consumption_kw"][-1], 92.5, places=6)
        self.assertAlmostEqual(res["naive_persistence_kw"], 92.5, places=6)
        self.assertTrue(
            any(w["code"] == "consumption_outside_training_range" for w in val["warnings"])
        )

    def test_random_scenario_generator_reproducibility_and_regimes(self) -> None:
        for district in SUPPORTED_DISTRICTS:
            for sc_key in SCENARIO_PRESETS:
                s1 = self.sim.generate_scenario(district, sc_key, seed=101)
                s2 = self.sim.generate_scenario(district, sc_key, seed=101)
                s3 = self.sim.generate_scenario(district, sc_key, seed=202)
                self.assertEqual(len(s1["consumption_kw"]), LOOKBACK_STEPS)
                self.assertEqual(s1["consumption_kw"], s2["consumption_kw"])
                self.assertNotEqual(s1["consumption_kw"], s3["consumption_kw"])

                # Paired pipeline weather must have 16x9 values and pass inference
                wx_retro = s1["paired_weather"]["consumption_weather_retro"]["rows"]
                self.assertEqual(len(wx_retro), LOOKBACK_STEPS)
                self.assertEqual(len(wx_retro[0]), 9)
                pred = self.sim.predict_next_interval(
                    "aggregate", district, "LSTM", "consumption_weather_retro",
                    s1["consumption_kw"], wx_retro
                )
                self.assertTrue(np.isfinite(pred["predicted_demand_kw"]))

            # Verify statistical ordering: low_demand < normal < high_demand
            low_m = np.mean([
                np.mean(self.sim.generate_scenario(district, "low_demand", seed=k)["consumption_kw"])
                for k in range(10)
            ])
            norm_m = np.mean([
                np.mean(self.sim.generate_scenario(district, "normal", seed=k)["consumption_kw"])
                for k in range(10)
            ])
            high_m = np.mean([
                np.mean(self.sim.generate_scenario(district, "high_demand", seed=k)["consumption_kw"])
                for k in range(10)
            ])
            self.assertLess(low_m, norm_m)
            self.assertLess(norm_m, high_m)

    def test_csv_input_consumption_and_weather(self) -> None:
        # 1. Consumption-only 20-row CSV with offset selection
        rows = ["timestamp,demand_target"]
        t0 = pd.Timestamp("2020-01-10 00:00:00")
        for i in range(20):
            ts_str = (t0 + pd.Timedelta(minutes=15 * i)).strftime("%Y-%m-%d %H:%M")
            rows.append(f"{ts_str},{14.0 + 0.25 * i:.3f}")
        csv_text = "\n".join(rows)

        parsed = self.sim.parse_csv_input(csv_text, "consumption_only", row_offset=4)
        self.assertEqual(len(parsed["consumption_kw"]), LOOKBACK_STEPS)
        self.assertAlmostEqual(parsed["consumption_kw"][0], 15.0)
        res = self.sim.predict_next_interval(
            "aggregate", "mathura", "LSTM", "consumption_only", parsed["consumption_kw"]
        )
        self.assertTrue(np.isfinite(res["predicted_demand_kw"]))

        # 2. Weather-augmented CSV from historical pipeline window
        win = self.sim.list_historical_weather_windows("bareilly", split="test", max_windows=2)[0]
        cols = FEATURE_SETS["consumption_weather_operational"]
        wx_lines = [",".join(cols)]
        for i in range(LOOKBACK_STEPS):
            vals = [win["consumption_kw"][i], *win["consumption_weather_operational"][i]]
            wx_lines.append(",".join(str(v) for v in vals))
        parsed_wx = self.sim.parse_csv_input(
            "\n".join(wx_lines), "consumption_weather_operational", row_offset=0
        )
        res_wx = self.sim.predict_next_interval(
            "aggregate", "bareilly", "1D_CNN", "consumption_weather_operational",
            parsed_wx["consumption_kw"], parsed_wx["weather_rows"]
        )
        self.assertTrue(np.isfinite(res_wx["predicted_demand_kw"]))

        # 3. Non-consecutive timestamp gap in CSV must be rejected
        gapped = list(rows)
        gapped[10] = "2020-01-10 05:00,16.500"
        with self.assertRaises(SimulatorValidationError):
            self.sim.parse_csv_input("\n".join(gapped), "consumption_only", row_offset=0)

        # 4. Duplicate timestamps in CSV must be explicitly rejected
        dup = list(rows)
        dup[5] = dup[4]
        with self.assertRaises(SimulatorValidationError) as ctx_dup:
            self.sim.parse_csv_input("\n".join(dup), "consumption_only", row_offset=0)
        self.assertIn("duplicate", str(ctx_dup.exception).lower())

        # 5. Unparseable timestamp string must be rejected
        bad_ts = list(rows)
        bad_ts[3] = "not-a-timestamp,15.200"
        with self.assertRaises(SimulatorValidationError):
            self.sim.parse_csv_input("\n".join(bad_ts), "consumption_only", row_offset=0)


if __name__ == "__main__":
    unittest.main()

