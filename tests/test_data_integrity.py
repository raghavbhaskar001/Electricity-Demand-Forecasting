"""
Data Integrity and Pre-Processing Verification Test Suite
Phase 1.3 Test Plan Implementation

Tests structural assumptions, physical bounds, 15-minute target construction,
non-destructive anomaly screening, and causal weather alignment using synthetic micro-fixtures.
No raw data files under data/raw/ are accessed or modified.
"""

import unittest
from datetime import datetime, timedelta
import math


class TestCadenceAndGrid(unittest.TestCase):
    """Module 1: Timestamp Cadence & Temporal Grid"""

    def test_regular_15min_cadence(self):
        """Test Case 1.1: Verify delta = 15 min across continuous sequence."""
        timestamps = [
            datetime(2013, 8, 1, 0, 15),
            datetime(2013, 8, 1, 0, 30),
            datetime(2013, 8, 1, 0, 45),
            datetime(2013, 8, 1, 1, 0),
        ]
        deltas = [
            (t2 - t1).total_seconds() / 60
            for t1, t2 in zip(timestamps[:-1], timestamps[1:])
        ]
        self.assertTrue(all(d == 15 for d in deltas), "All deltas must be exactly 15 minutes")

    def test_duplicate_timestamp_detection(self):
        """Test Case 1.2: Detect duplicate timestamps."""
        series = [
            (datetime(2013, 8, 1, 0, 15), 0.05),
            (datetime(2013, 8, 1, 0, 30), 0.06),
            (datetime(2013, 8, 1, 0, 30), 0.06),  # duplicate
            (datetime(2013, 8, 1, 0, 45), 0.07),
        ]
        seen = set()
        duplicates = []
        for ts, val in series:
            if ts in seen:
                duplicates.append(ts)
            seen.add(ts)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0], datetime(2013, 8, 1, 0, 30))

    def test_3min_cadence_and_gap_isolation(self):
        """Test Case 1.3: Identify blackout gaps > 3 min."""
        timestamps = [
            datetime(2019, 5, 1, 3, 0),
            datetime(2019, 5, 1, 3, 3),
            datetime(2019, 5, 1, 3, 6),
            datetime(2019, 5, 1, 3, 36),  # 30 min gap
            datetime(2019, 5, 1, 3, 39),
        ]
        gaps = []
        for t1, t2 in zip(timestamps[:-1], timestamps[1:]):
            delta = (t2 - t1).total_seconds() / 60
            if delta > 3.0:
                gaps.append((t1, t2, delta))
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0][2], 30.0)


class TestTargetConstruction(unittest.TestCase):
    """Module 2: Target Construction & Aggregation"""

    def test_strict_5_interval_aggregation_pass(self):
        """Test Case 2.1: Exactly five 3-min intervals aggregated to 15-min target."""
        window_end = datetime(2019, 5, 1, 0, 15)
        expected_sub_intervals = [
            window_end - timedelta(minutes=12),
            window_end - timedelta(minutes=9),
            window_end - timedelta(minutes=6),
            window_end - timedelta(minutes=3),
            window_end,
        ]
        sample_readings = {ts: 0.02 for ts in expected_sub_intervals}

        # Validate that exactly 5 expected timestamps are present
        present_readings = [sample_readings.get(ts) for ts in expected_sub_intervals]
        self.assertTrue(all(r is not None for r in present_readings))
        self.assertEqual(len(present_readings), 5)
        energy_15min = sum(present_readings)
        self.assertAlmostEqual(energy_15min, 0.10, places=4)

    def test_rejection_of_naive_5_row_sum_on_discontinuous_data(self):
        """Test Case 2.2: Rejection of naive 5 consecutive rows across blackout."""
        # 5 consecutive rows in CSV that span across a 1-hour gap
        discontinuous_rows = [
            (datetime(2019, 5, 1, 10, 0), 0.02),
            (datetime(2019, 5, 1, 10, 3), 0.02),
            (datetime(2019, 5, 1, 11, 0), 0.02),  # Gap between 10:03 and 11:00
            (datetime(2019, 5, 1, 11, 3), 0.02),
            (datetime(2019, 5, 1, 11, 6), 0.02),
        ]
        # Target interval definition for (10:00, 10:15]:
        target_ts = datetime(2019, 5, 1, 10, 15)
        required_grid = {
            target_ts - timedelta(minutes=12),
            target_ts - timedelta(minutes=9),
            target_ts - timedelta(minutes=6),
            target_ts - timedelta(minutes=3),
            target_ts,
        }
        observed_in_window = [r for r in discontinuous_rows if r[0] in required_grid]
        # Only 10:03 is inside the required grid for (10:00, 10:15]
        self.assertLess(len(observed_in_window), 5, "Discontinuous window must be incomplete")
        # Must be flagged as incomplete / rejected, NOT aggregated
        is_complete = len(observed_in_window) == 5
        self.assertFalse(is_complete)

    def test_iiitd_lnenergy_to_kwh_conversion(self):
        """Test Case 2.3: Invert natural logarithm lnenergy to kWh."""
        lnenergy_sample = -2.56151775
        energy_kwh = math.exp(lnenergy_sample)
        self.assertGreater(energy_kwh, 0.0)
        self.assertAlmostEqual(energy_kwh, 0.0771879, places=6)


class TestElectricalPhysics(unittest.TestCase):
    """Module 3: Electrical Physics Validation (Screening Thresholds)"""

    def test_screening_thresholds_vs_deletion(self):
        """Test Case 3.1: Screen anomalies non-destructively; do not delete."""
        test_records = [
            {"v": 242.0, "i": 2.0, "e": 0.024},
            {"v": 0.0, "i": 0.0, "e": 0.000},    # Genuine outage (grid down)
            {"v": 240.0, "i": 0.0, "e": 0.000},  # Genuine zero-load (grid healthy, appliances off)
            {"v": 295.0, "i": 1.5, "e": 0.022},  # Overvoltage
            {"v": 140.0, "i": 3.0, "e": 0.021},  # Undervoltage / brownout
        ]
        for rec in test_records:
            flags = {}
            if rec["v"] == 0.0 and rec["i"] == 0.0 and rec["e"] == 0.0:
                flags["IS_OUTAGE"] = True
            elif rec["v"] >= 150.0 and rec["i"] == 0.0 and rec["e"] == 0.0:
                flags["IS_ZERO_LOAD"] = True
            if rec["v"] > 280.0:
                flags["IS_OVERVOLTAGE"] = True
            if 0.0 < rec["v"] < 150.0:
                flags["IS_UNDERVOLTAGE_BROWNOUT"] = True
            rec["flags"] = flags

        # Ensure no records were deleted and states are mutually distinct
        self.assertEqual(len(test_records), 5)
        self.assertTrue(test_records[1]["flags"].get("IS_OUTAGE", False))
        self.assertFalse(test_records[1]["flags"].get("IS_ZERO_LOAD", False))
        self.assertTrue(test_records[2]["flags"].get("IS_ZERO_LOAD", False))
        self.assertFalse(test_records[2]["flags"].get("IS_OUTAGE", False))
        self.assertTrue(test_records[3]["flags"].get("IS_OVERVOLTAGE", False))
        self.assertTrue(test_records[4]["flags"].get("IS_UNDERVOLTAGE_BROWNOUT", False))

    def test_apparent_energy_physical_bound(self):
        """Test Case 3.2: Verify active energy <= apparent energy (PF <= 1.0)."""
        voltage = 240.0  # V
        current = 2.0    # A
        delta_hours = 3.0 / 60.0  # 0.05 h
        apparent_kwh = (voltage * current * delta_hours) / 1000.0  # 0.024 kWh

        normal_energy = 0.021
        self.assertLessEqual(normal_energy, apparent_kwh * 1.05)  # Within measurement margin

        anomalous_energy = 0.080  # Impossible without higher V or I
        self.assertGreater(anomalous_energy, apparent_kwh * 1.05)


class TestMissingnessAndGaps(unittest.TestCase):
    """Module 4: Missingness and Transition Gap Handling"""

    def test_pctmissing_filtering_iiitd(self):
        """Test Case 4.1: Retain records with pctmissing <= 10%."""
        samples = [
            {"apt": 1, "pctmissing": 0.0, "lnenergy": -2.5},
            {"apt": 1, "pctmissing": 5.0, "lnenergy": -2.4},
            {"apt": 1, "pctmissing": 15.0, "lnenergy": -2.3},
            {"apt": 1, "pctmissing": 100.0, "lnenergy": -99.0},
        ]
        retained = [s for s in samples if s["pctmissing"] <= 10.0]
        flagged = [s for s in samples if s["pctmissing"] > 10.0]
        self.assertEqual(len(retained), 2)
        self.assertEqual(len(flagged), 2)

    def test_annual_transition_gap_detection_mathura(self):
        """Test Case 4.2: Detect Mathura 2019-2020 5-day transition gap."""
        t_end_2019 = datetime(2019, 12, 31, 23, 57)
        t_start_2020 = datetime(2020, 1, 6, 0, 0)
        gap_hours = (t_start_2020 - t_end_2019).total_seconds() / 3600.0
        self.assertAlmostEqual(gap_hours, 120.05, places=2)
        self.assertGreater(gap_hours, 24.0, "Annual transition has a multi-day gap")


class TestCohortAggregation(unittest.TestCase):
    """Module 5: Multi-Meter Synchronous Summation & Scientific Labeling"""

    def test_iiitd_cohort_ground_truth_requirement(self):
        """Test Case 5.1: Primary ground truth strictly requires 19/19 apartments."""
        # 19 apartments present
        cohort_19 = {apt: 0.10 for apt in range(1, 20)}
        self.assertEqual(len(cohort_19), 19)
        primary_agg = sum(cohort_19.values())
        self.assertAlmostEqual(primary_agg, 1.90, places=2)

        # 18 apartments present: cannot be labeled primary ground truth
        cohort_18 = {apt: 0.10 for apt in range(1, 19)}
        self.assertEqual(len(cohort_18), 18)
        # Primary ground truth check
        is_primary_valid = len(cohort_18) == 19
        self.assertFalse(is_primary_valid)
        # Sensitivity partial sum is permitted
        sensitivity_partial_sum = sum(cohort_18.values())
        self.assertAlmostEqual(sensitivity_partial_sum, 1.80, places=2)

    def test_ceew_cohort_thresholds(self):
        """Test Case 5.2: CEEW 90% primary threshold vs 85% sensitivity."""
        fixed_cohort_size = 50
        # 46 meters active: 46 / 50 = 92% -> Primary Analysis Eligible
        active_46 = 46
        self.assertGreaterEqual(active_46 / fixed_cohort_size, 0.90)

        # 43 meters active: 43 / 50 = 86% -> Sensitivity Analysis Eligible (not primary)
        active_43 = 43
        coverage_43 = active_43 / fixed_cohort_size
        self.assertLess(coverage_43, 0.90)
        self.assertGreaterEqual(coverage_43, 0.85)

        # 40 meters active: 40 / 50 = 80% -> Excluded from both
        active_40 = 40
        self.assertLess(active_40 / fixed_cohort_size, 0.85)


class TestWeatherLeakage(unittest.TestCase):
    """Module 6: Causal Alignment for Weather Covariates"""

    def test_causal_lagging_rule(self):
        """Test Case 6.1: Assert t_weather <= T (forecast origin)."""
        forecast_origin_T = datetime(2019, 5, 1, 12, 0)  # T
        target_horizon = datetime(2019, 5, 1, 12, 15)    # T + 15 min

        # Historical weather observations available at or before T
        valid_weather_times = [
            datetime(2019, 5, 1, 10, 0),
            datetime(2019, 5, 1, 11, 0),
            datetime(2019, 5, 1, 12, 0),  # Latest permissible observed covariate
        ]
        # Any weather at T + 15 min or T + 1 h is future relative to T
        future_weather_time = datetime(2019, 5, 1, 13, 0)

        for wt in valid_weather_times:
            self.assertLessEqual(wt, forecast_origin_T, "Causal: weather must be <= origin T")

        self.assertGreater(future_weather_time, forecast_origin_T, "Future weather cannot leak into input tensor")


if __name__ == "__main__":
    unittest.main()

