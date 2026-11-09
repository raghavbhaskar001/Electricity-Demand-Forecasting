r"""
Baseline Forecasting Models Module
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models

Implements:
1. Naive Persistence Baseline: \hat{y}_{t+1} = y_t
2. Seasonal Same-Time Previous-Day Persistence Baseline: \hat{y}_{t+1} = y_{t+1-96} (24-hour lag = 96 steps of 15 min)

Computes benchmark metrics on identical test partitions as deep learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from src.dataset import extract_contiguous_sequences, temporal_train_val_test_split
from src.evaluate import compute_forecasting_metrics


class NaivePersistence:
    r"""
    Naive Persistence Forecaster: \hat{y}_{t+1} = y_t
    Predicts that the next 15-minute interval demand will equal the current 15-minute interval demand.
    """
    def __init__(self):
        self.name = "Naive_Persistence"

    def predict_from_sequences(self, X: np.ndarray, target_feature_idx: int = 0) -> np.ndarray:
        """
        Extracts the last observed time-step of the target feature from the lookback sequence.
        Args:
            X: Array of shape (N, L, F) where L is lookback (16) and F is features.
            target_feature_idx: Index of demand target feature in F (default: 0).
        Returns:
            Array of shape (N, 1) containing y_t predictions for y_{t+1}.
        """
        assert len(X.shape) == 3, f"Expected 3D input (N, L, F), got {X.shape}"
        return X[:, -1, target_feature_idx:target_feature_idx+1].astype(np.float64)


class SeasonalPersistence:
    r"""
    Seasonal Same-Time Previous-Day Persistence Forecaster: \hat{y}_{t+1} = y_{t+1-96}
    Predicts that demand will equal the demand observed at the exact same 15-minute time of day 24 hours prior.
    """
    def __init__(self, seasonal_lag: int = 96):
        self.name = "Seasonal_Previous_Day_Persistence"
        self.seasonal_lag = seasonal_lag  # 96 steps = 24 hours at 15-minute intervals

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        target_timestamps: np.ndarray,
        target_col: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Looks up the observed demand 24 hours prior to each target timestamp.
        Args:
            df: DataFrame containing 'timestamp' and target_col.
            target_timestamps: Array of target timestamps for test sequence targets.
            target_col: Name of demand target column.
        Returns:
            y_pred: Predictions array of shape (N, 1).
            valid_mask: Boolean mask indicating which predictions had valid 24h lag observations.
        """
        # Build lookup series indexed by timestamp
        ts_index = pd.to_datetime(df["timestamp"])
        lookup_s = pd.Series(df[target_col].values, index=ts_index)

        target_dt = pd.to_datetime(target_timestamps)
        lagged_dt = target_dt - pd.Timedelta(days=1)

        lagged_values = lookup_s.reindex(lagged_dt).values
        valid_mask = ~np.isnan(lagged_values)

        return lagged_values.reshape(-1, 1), valid_mask


def evaluate_baselines_on_dataset(
    features_df: pd.DataFrame,
    target_col: str,
    lookback_len: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge_gap: int = 16
) -> Dict[str, Dict]:
    """
    Extracts contiguous sequences, applies chronological train/val/test split with purge gap,
    and evaluates both Naive Persistence and Seasonal Persistence on the test partition.
    """
    X, y, ts = extract_contiguous_sequences(
        features_df,
        target_col=target_col,
        feature_cols=[target_col],
        lookback_len=lookback_len,
        horizon_len=1
    )

    if len(X) == 0:
        return {"error": "No contiguous sequences found"}

    # Apply chronological split with boundary purge gap
    (X_tr, y_tr, ts_tr), (X_va, y_va, ts_va), (X_te, y_te, ts_te) = temporal_train_val_test_split(
        X, y, ts,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        purge_gap=purge_gap
    )

    if len(y_te) == 0:
        return {
            "census": {
                "total_sequences": len(X),
                "train_sequences": len(X_tr),
                "val_sequences": len(X_va),
                "test_sequences": 0,
                "note": "Insufficient sequence count for test partition after boundary purge"
            }
        }

    # 1. Full test naive persistence
    naive = NaivePersistence()
    y_pred_naive = naive.predict_from_sequences(X_te, target_feature_idx=0)
    naive_full_metrics = compute_forecasting_metrics(
        y_true=y_te,
        y_pred=y_pred_naive,
        y_persistence=y_pred_naive  # Baseline vs itself
    )

    # 2. Seasonal Persistence (same-time previous day)
    seasonal = SeasonalPersistence(seasonal_lag=96)
    y_pred_seas, valid_seas_mask = seasonal.predict_from_dataframe(
        df=features_df,
        target_timestamps=ts_te,
        target_col=target_col
    )

    # 3. Matched subset evaluation on exact identical timestamps
    matched_eval = {}
    if valid_seas_mask.sum() > 0:
        y_te_matched = y_te[valid_seas_mask]
        y_naive_matched = y_pred_naive[valid_seas_mask]
        y_seas_matched = y_pred_seas[valid_seas_mask]

        # Naive on matched subset
        naive_matched_metrics = compute_forecasting_metrics(
            y_true=y_te_matched,
            y_pred=y_naive_matched,
            y_persistence=y_naive_matched
        )

        # Seasonal on matched subset, with skill score strictly relative to naive on matched subset
        seas_matched_metrics = compute_forecasting_metrics(
            y_true=y_te_matched,
            y_pred=y_seas_matched,
            y_persistence=y_naive_matched
        )

        matched_eval = {
            "matched_sample_count": int(valid_seas_mask.sum()),
            "coverage_ratio": round(float(valid_seas_mask.mean()), 4),
            "naive_persistence_matched": naive_matched_metrics,
            "seasonal_persistence_matched": seas_matched_metrics,
            "seasonal_skill_score_vs_naive": seas_matched_metrics["persistence_skill_score"],
        }
    else:
        matched_eval = {
            "error": "No valid 24h prior observations available in test period",
            "matched_sample_count": 0,
            "coverage_ratio": 0.0
        }

    return {
        "census": {
            "total_sequences": len(X),
            "train_sequences": len(X_tr),
            "val_sequences": len(X_va),
            "test_sequences": len(y_te),
            "test_start_ts": str(ts_te[0]),
            "test_end_ts": str(ts_te[-1]),
        },
        "full_test_evaluation": {
            "test_sample_count": len(y_te),
            "naive_persistence": naive_full_metrics,
            "seasonal_coverage_ratio": round(float(valid_seas_mask.mean()), 4),
            "seasonal_valid_count": int(valid_seas_mask.sum()),
            "seasonal_missing_count": int(len(y_te) - valid_seas_mask.sum()),
        },
        "matched_subset_evaluation": matched_eval,
    }


def evaluate_individual_meters(
    parquet_path: str,
    lookback_len: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge_gap: int = 16
) -> Dict:
    """
    Evaluates Naive Persistence baseline across all individual smart meters.
    Computes both pooled metrics and distribution across meters (median, IQR, min, max).
    """
    ind_df = pd.read_parquet(parquet_path)
    meter_cols = [c for c in ind_df.columns if c != "timestamp"]

    all_y_te = []
    all_y_pred = []
    meter_metrics_list = []
    
    total_seqs_count = 0
    total_tr_count = 0
    total_va_count = 0
    total_te_count = 0
    total_purge_count = 0

    for m in meter_cols:
        X, y, ts = extract_contiguous_sequences(
            ind_df,
            target_col=m,
            feature_cols=[m],
            lookback_len=lookback_len,
            horizon_len=1
        )
        if len(X) < 100:
            continue
        (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, _) = temporal_train_val_test_split(
            X, y, ts,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            purge_gap=purge_gap
        )
        if len(y_te) == 0:
            continue

        total_seqs_count += len(X)
        total_tr_count += len(X_tr)
        total_va_count += len(X_va)
        total_te_count += len(X_te)
        total_purge_count += (2 * purge_gap)

        naive = NaivePersistence()
        y_pred = naive.predict_from_sequences(X_te, target_feature_idx=0)
        m_metric = compute_forecasting_metrics(y_te, y_pred, y_persistence=y_pred)
        m_metric["meter_id"] = m
        m_metric["test_sample_count"] = len(y_te)
        meter_metrics_list.append(m_metric)

        all_y_te.append(y_te)
        all_y_pred.append(y_pred)

    if not all_y_te:
        return {"error": "No valid meter sequences"}

    y_te_pool = np.vstack(all_y_te)
    y_pred_pool = np.vstack(all_y_pred)
    pooled_metrics = compute_forecasting_metrics(y_te_pool, y_pred_pool, y_persistence=y_pred_pool)
    pooled_metrics["total_test_sequences"] = len(y_te_pool)
    pooled_metrics["evaluated_meter_count"] = len(meter_metrics_list)

    # Compute distribution statistics across individual meters
    rmses = [m["rmse_kw"] for m in meter_metrics_list if m["rmse_kw"] is not None]
    maes = [m["mae_kw"] for m in meter_metrics_list if m["mae_kw"] is not None]
    wapes = [m["wape_pct"] for m in meter_metrics_list if m["wape_pct"] is not None]
    r2s = [m["r2_score"] for m in meter_metrics_list if m["r2_score"] is not None]

    meter_distribution = {
        "rmse_kw": {
            "median": round(float(np.median(rmses)), 4),
            "iqr": round(float(np.percentile(rmses, 75) - np.percentile(rmses, 25)), 4),
            "min": round(float(np.min(rmses)), 4),
            "max": round(float(np.max(rmses)), 4),
        },
        "mae_kw": {
            "median": round(float(np.median(maes)), 4),
            "iqr": round(float(np.percentile(maes, 75) - np.percentile(maes, 25)), 4),
            "min": round(float(np.min(maes)), 4),
            "max": round(float(np.max(maes)), 4),
        },
        "wape_pct": {
            "median": round(float(np.median(wapes)), 2),
            "iqr": round(float(np.percentile(wapes, 75) - np.percentile(wapes, 25)), 2),
            "min": round(float(np.min(wapes)), 2),
            "max": round(float(np.max(wapes)), 2),
        },
        "r2_score": {
            "median": round(float(np.median(r2s)), 4),
            "iqr": round(float(np.percentile(r2s, 75) - np.percentile(r2s, 25)), 4),
            "min": round(float(np.min(r2s)), 4),
            "max": round(float(np.max(r2s)), 4),
        }
    }

    return {
        "census": {
            "total_sequences": total_seqs_count,
            "train_sequences": total_tr_count,
            "val_sequences": total_va_count,
            "test_sequences": total_te_count,
            "purge_sequences": total_purge_count,
            "reconciled_sum": total_tr_count + total_va_count + total_te_count + total_purge_count,
            "evaluated_meters": len(meter_metrics_list),
        },
        "pooled": pooled_metrics,
        "meter_distribution": meter_distribution,
        "meter_count": len(meter_metrics_list),
    }


def run_all_baseline_benchmarks(output_json_path: Optional[str] = "artifacts/metrics/baseline_benchmarks.json") -> Dict:
    """
    Executes baseline benchmarking across all experimental datasets and cohorts.
    Saves results to JSON.
    """
    import os
    import json

    benchmarks = {
        "metadata": {
            "project": "Electricity Demand Forecasting Using Deep Learning",
            "study_scale": ["aggregate_cohort", "individual_meter"],
            "lookback_len": 16,
            "lookback_hours": 4.0,
            "horizon_steps": 1,
            "horizon_minutes": 15,
            "train_val_test_split": "70/15/15 chronological with L=16 boundary purge gap"
        },
        "aggregates": {},
        "individual_meters": {}
    }

    # 1. Mathura Aggregate
    print("Evaluating Mathura Aggregate Baselines...")
    mathura_feat = pd.read_parquet("data/processed/mathura_features.parquet")
    benchmarks["aggregates"]["mathura"] = evaluate_baselines_on_dataset(
        mathura_feat, target_col="aggregate_demand_kw_primary"
    )

    # 2. Bareilly Aggregate
    print("Evaluating Bareilly Aggregate Baselines...")
    bareilly_feat = pd.read_parquet("data/processed/bareilly_features.parquet")
    benchmarks["aggregates"]["bareilly"] = evaluate_baselines_on_dataset(
        bareilly_feat, target_col="aggregate_demand_kw_primary"
    )

    # 3. Delhi Aggregate (Feasibility Census)
    print("Evaluating Delhi Aggregate Feasibility...")
    delhi_feat = pd.read_parquet("data/processed/delhi_features.parquet")
    benchmarks["aggregates"]["delhi_exploratory"] = evaluate_baselines_on_dataset(
        delhi_feat, target_col="aggregate_demand_kw_primary"
    )

    # 4. Mathura Individual Meters
    print("Evaluating Mathura Individual Meters Baselines...")
    benchmarks["individual_meters"]["mathura"] = evaluate_individual_meters(
        "data/processed/mathura_individual_meters.parquet"
    )

    # 5. Bareilly Individual Meters
    print("Evaluating Bareilly Individual Meters Baselines...")
    benchmarks["individual_meters"]["bareilly"] = evaluate_individual_meters(
        "data/processed/bareilly_individual_meters.parquet"
    )

    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(benchmarks, f, indent=2)
        print(f"Saved benchmark outputs to {output_json_path}")

    return benchmarks


if __name__ == "__main__":
    run_all_baseline_benchmarks()
