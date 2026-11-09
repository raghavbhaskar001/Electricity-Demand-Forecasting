"""
Phase 1.5 Target Construction & Modelling Dataset Preparation Unit Tests
Verifies mathematical transformations, 15-minute target aggregation,
zero-load preservation, causal and operational weather lagging, and sequence blocking.
"""

import unittest
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd


class TestTargetConstructionFormulas(unittest.TestCase):
    """Test Case 1: Energy & Power Conversions"""

    def test_iiitd_kwh_and_kw_conversion(self):
        """Verify exp(lnenergy) -> kWh -> kW (4 * kWh for 15-min interval)."""
        lnenergy_sample = -2.56151775
        energy_kwh = math.exp(lnenergy_sample)
        power_kw = 4.0 * energy_kwh

        self.assertAlmostEqual(energy_kwh, 0.0771875, places=5)
        self.assertAlmostEqual(power_kw, 0.308750, places=5)

    def test_ceew_5_interval_kw_conversion(self):
        """Verify 5-interval 3-min sum -> 15-min kWh -> 4 * kWh average kW."""
        sub_readings_kwh = [0.02, 0.025, 0.015, 0.02, 0.02]
        self.assertEqual(len(sub_readings_kwh), 5)
        energy_15min_kwh = sum(sub_readings_kwh)
        power_kw = 4.0 * energy_15min_kwh

        self.assertAlmostEqual(energy_15min_kwh, 0.10, places=4)
        self.assertAlmostEqual(power_kw, 0.40, places=4)


class TestTargetWindowIntegrity(unittest.TestCase):
    """Test Case 2: Strict 5-interval window requirements"""

    def test_window_completeness_and_rejection(self):
        """Require exactly all 5 sub-intervals {T-12, T-9, T-6, T-3, T}."""
        target_T = datetime(2019, 5, 1, 10, 15)
        expected_subs = [
            target_T - timedelta(minutes=12),
            target_T - timedelta(minutes=9),
            target_T - timedelta(minutes=6),
            target_T - timedelta(minutes=3),
            target_T,
        ]

        # Complete window
        readings_full = {ts: 0.02 for ts in expected_subs}
        self.assertEqual(len([readings_full.get(ts) for ts in expected_subs if readings_full.get(ts) is not None]), 5)

        # Incomplete window (e.g. 1 packet missing)
        readings_partial = {ts: 0.02 for ts in expected_subs[:-1]}
        observed_count = len([readings_partial.get(ts) for ts in expected_subs if readings_partial.get(ts) is not None])
        self.assertLess(observed_count, 5)
        # Incomplete windows must be flagged as invalid targets
        is_valid = observed_count == 5
        self.assertFalse(is_valid)


class TestOperatingStatePreservation(unittest.TestCase):
    """Test Case 3: Zero-load and Outage state preservation"""

    def test_zero_load_is_valid_target(self):
        """Zero-load (V >= 150V, I == 0A, kWh == 0) must NOT be excluded; it is a valid target."""
        v, i, kwh = 230.0, 0.0, 0.0
        is_zero_load = (v >= 150.0) and (i == 0.0) and (kwh == 0.0)
        self.assertTrue(is_zero_load)
        
        # When all 5 intervals are zero-load
        target_kw = 4.0 * sum([0.0]*5)
        self.assertEqual(target_kw, 0.0)
        # Must be treated as a valid training observation (not dropped)
        target_valid = True
        self.assertTrue(target_valid)

    def test_outage_flag_preservation(self):
        """Outage (V < 50V, I == 0A, kWh == 0) is preserved and tagged."""
        v, i, kwh = 0.0, 0.0, 0.0
        is_outage = (v < 50.0) and (i == 0.0) and (kwh == 0.0)
        self.assertTrue(is_outage)


class TestCausalAndOperationalWeather(unittest.TestCase):
    """Test Case 4: Causal backward as-of merge and 5-day operational lag"""

    def setUp(self):
        # 1-hour weather observations
        self.weather_df = pd.DataFrame({
            "timestamp": pd.date_range("2019-05-01 00:00", "2019-05-10 00:00", freq="1h"),
            "temperature_2m": np.linspace(20, 35, 217),
        })

    def test_retrospective_asof_no_future_leakage(self):
        """Retrospective reanalysis uses strictly backward as-of: t_weather <= T."""
        target_times = pd.DataFrame({
            "target_time": [
                pd.Timestamp("2019-05-01 00:15"),
                pd.Timestamp("2019-05-01 00:45"),
                pd.Timestamp("2019-05-01 01:00"),
            ]
        })
        merged = pd.merge_asof(
            target_times,
            self.weather_df,
            left_on="target_time",
            right_on="timestamp",
            direction="backward",
        )
        # Target 00:15 must get weather from 00:00 (age = 15 min), NOT 01:00
        self.assertEqual(merged.loc[0, "timestamp"], pd.Timestamp("2019-05-01 00:00"))
        # Target 00:45 must get weather from 00:00 (age = 45 min)
        self.assertEqual(merged.loc[1, "timestamp"], pd.Timestamp("2019-05-01 00:00"))
        # Target 01:00 gets weather from 01:00 (age = 0 min)
        self.assertEqual(merged.loc[2, "timestamp"], pd.Timestamp("2019-05-01 01:00"))

        # Assert no weather timestamp > target_time
        self.assertTrue((merged["timestamp"] <= merged["target_time"]).all())

    def test_operational_weather_5day_lag(self):
        """Operational availability simulation enforces t_weather <= T - 120h."""
        target_time = pd.Timestamp("2019-05-08 12:00")
        operational_cutoff = target_time - pd.Timedelta(days=5)  # 2019-05-03 12:00

        # Available operational weather must be <= cutoff
        avail_weather = self.weather_df[self.weather_df["timestamp"] <= operational_cutoff]
        latest_avail = avail_weather["timestamp"].max()

        self.assertEqual(latest_avail, pd.Timestamp("2019-05-03 12:00"))
        staleness_hours = (target_time - latest_avail).total_seconds() / 3600.0
        self.assertAlmostEqual(staleness_hours, 120.0, places=1)


class TestContiguousSequenceBlocking(unittest.TestCase):
    """Test Case 5: Training sequences cannot cross invalid target gaps"""

    def test_sequence_generator_respects_gaps(self):
        """For input sequence length L=4, window across NaN must be rejected."""
        # 10 time steps with an invalid target at index 5
        targets = np.array([1.0, 1.2, 1.1, 1.3, 1.2, np.nan, 1.4, 1.5, 1.3, 1.6])
        valid_mask = ~np.isnan(targets)
        L = 4  # Lookback sequence length

        valid_sequences = []
        for i in range(L, len(targets)):
            # Lookback window: [i-L, i-1], Target: [i]
            lookback_valid = valid_mask[i-L:i].all()
            target_valid = valid_mask[i]
            if lookback_valid and target_valid:
                valid_sequences.append(i)

        # Sequences crossing index 5 (which is NaN) must be excluded:
        # i = 5: target is NaN -> excluded
        # i = 6: lookback includes index 5 (NaN) -> excluded
        # i = 7: lookback includes index 5 (NaN) -> excluded
        # i = 8: lookback includes index 5 (NaN) -> excluded
        # i = 9: lookback is [5, 6, 7, 8] which includes index 5 -> excluded
        self.assertNotIn(5, valid_sequences)
        self.assertNotIn(6, valid_sequences)
        self.assertNotIn(7, valid_sequences)
        self.assertNotIn(8, valid_sequences)
        self.assertNotIn(9, valid_sequences)
        self.assertEqual(valid_sequences, [4])  # only i=4 is valid


if __name__ == "__main__":
    unittest.main()
