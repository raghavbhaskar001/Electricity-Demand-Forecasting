"""
Interactive Electricity Demand Prediction Simulator Engine.

Provides scientifically audited next-15-minute (H=1) inference from 16 consecutive
15-minute intervals (L=16, 4 hours) using the trained PyTorch checkpoints and
exact training-split StandardScalers from the research pipeline.

Integrity guarantees:
  - Builds an explicit 36-combination availability matrix before running inference.
  - Supports ONLY combinations with verified finalized checkpoints and matching
    preprocessing scalers (18/18 aggregate feeder models).
  - Explicitly rejects unsupported individual-household models (including the
    interrupted epoch-3 partial checkpoint `ind_mathura_1D_CNN_consumption_only.pt`)
    and never substitutes aggregate checkpoints for individual inference.
  - Preserves exact `FEATURE_SETS` column order and 70/15/15 (purge_gap=16)
    training-only `StandardScaler` parameters.
  - Validates 16 consecutive numeric inputs, flags out-of-distribution (OOD)
    observations against training ranges without silently clipping values, and
    never fabricates weather features.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.dataset import (
    FEATURE_SETS,
    extract_contiguous_sequences,
    temporal_train_val_test_split,
)
from src.train_runner import StandardScaler, get_model

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CHECKPOINTS_DIR = ROOT_DIR / "artifacts" / "checkpoints"
METRICS_DIR = ROOT_DIR / "artifacts" / "metrics"

LOOKBACK_STEPS: int = 16
HORIZON_STEPS: int = 1
INTERVAL_MINUTES: int = 15

SUPPORTED_SCALES: Tuple[str, ...] = ("aggregate", "individual")
SUPPORTED_DISTRICTS: Tuple[str, ...] = ("mathura", "bareilly")
SUPPORTED_ARCHITECTURES: Tuple[str, ...] = ("1D_CNN", "LSTM", "CNN_LSTM")
SUPPORTED_FEATURE_CONFIGS: Tuple[str, ...] = (
    "consumption_only",
    "consumption_weather_retro",
    "consumption_weather_operational",
)

SCENARIO_PRESETS: Dict[str, Dict[str, str]] = {
    "normal": {
        "label": "Normal Consumption",
        "description": "Typical mid-quantile feeder load trajectory (35th–65th training percentile) with moderate step-to-step variability.",
    },
    "high_demand": {
        "label": "High Demand",
        "description": "Sustained upper-decile feeder demand (>= 90th training percentile) sampled from peak thermal load periods.",
    },
    "low_demand": {
        "label": "Low Demand",
        "description": "Lower-decile baseload trajectory (<= 10th training percentile) sampled from off-peak periods.",
    },
    "morning_peak": {
        "label": "Morning Peak",
        "description": "4-hour morning activity trajectory ending between 06:30 and 10:45 IST sampled from above-median morning windows.",
    },
    "evening_peak": {
        "label": "Evening Peak",
        "description": "4-hour evening ramp/peak trajectory ending between 18:30 and 22:45 IST sampled from domestic lighting/cooling surges.",
    },
    "irregular": {
        "label": "Irregular Consumption",
        "description": "High-volatility feeder trajectory (>= 90th percentile of 15-minute absolute step changes) reflecting transient load swings.",
    },
}

CONSUMPTION_COLUMN_ALIASES: Tuple[str, ...] = (
    "demand_target",
    "consumption_kw",
    "aggregate_demand_kw_primary",
    "actual_demand_kw",
    "demand_kw",
    "kw",
)

WEATHER_UNITS: Dict[str, str] = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "dew_point_2m": "°C",
    "surface_pressure": "hPa",
    "wind_speed_10m": "km/h",
    "direct_radiation": "W/m²",
    "diffuse_radiation": "W/m²",
    "cloud_cover": "%",
    "precipitation": "mm",
}


class SimulatorValidationError(ValueError):
    """Raised when user inputs violate simulator structural or physical rules."""


class UnsupportedConfigurationError(RuntimeError):
    """Raised when a requested scale/district/architecture/feature combination cannot run genuine inference."""


def _sha256_short(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def make_experiment_id(scale: str, district: str, arch: str, feature_config: str) -> str:
    prefix = "agg" if scale == "aggregate" else "ind"
    return f"{prefix}_{district}_{arch}_{feature_config}"


def weather_feature_names(feature_config: str) -> List[str]:
    cols = FEATURE_SETS.get(feature_config, [])
    return [c for c in cols if c != "demand_target"]


def _weather_physical_bounds_error(col_name: str, val: float) -> Optional[str]:
    """Check non-negotiable physical bounds (without clipping unusual weather)."""
    base = col_name.replace("retro_", "").replace("operational_", "")
    if base in ("relative_humidity_2m", "cloud_cover") and (val < 0.0 or val > 100.0):
        return f"{col_name}={val} outside physical [0, 100]% bounds"
    if base in ("wind_speed_10m", "direct_radiation", "diffuse_radiation", "precipitation") and val < 0.0:
        return f"{col_name}={val} cannot be negative"
    if base == "surface_pressure" and (val <= 500.0 or val >= 1150.0):
        return f"{col_name}={val} hPa outside physical atmospheric bounds (500–1150 hPa)"
    if base in ("temperature_2m", "dew_point_2m") and (val < -50.0 or val > 70.0):
        return f"{col_name}={val} °C outside physical terrestrial bounds (-50 to 70 °C)"
    return None


@dataclass
class DistrictTrainingContext:
    district: str
    # Raw 16-step training windows in physical units using the 19-feature union or per-config arrays
    X_train_by_config: Dict[str, np.ndarray]
    y_train_by_config: Dict[str, np.ndarray]
    ts_train_by_config: Dict[str, np.ndarray]
    X_test_by_config: Dict[str, np.ndarray]
    y_test_by_config: Dict[str, np.ndarray]
    ts_test_by_config: Dict[str, np.ndarray]
    feat_scalers: Dict[str, StandardScaler]
    targ_scalers: Dict[str, StandardScaler]
    feature_stats: Dict[str, Dict[str, Dict[str, float]]]
    regime_indices: Dict[str, np.ndarray]
    regime_chol: Dict[str, np.ndarray]


class PredictionSimulator:
    """
    Stateful, cached inference and scenario engine backed by verified project artifacts.
    """

    def __init__(self, root_dir: Path = ROOT_DIR) -> None:
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / "data" / "processed"
        self.checkpoints_dir = self.root_dir / "artifacts" / "checkpoints"
        self.metrics_dir = self.root_dir / "artifacts" / "metrics"

        self._training_records: Dict[str, Dict[str, Any]] = self._load_training_records()
        self._manifest_types: Dict[str, str] = self._load_manifest_types()
        self._availability: List[Dict[str, Any]] = self._build_availability_matrix()
        self._availability_by_id: Dict[str, Dict[str, Any]] = {
            row["exp_id"]: row for row in self._availability
        }
        self._contexts: Dict[str, DistrictTrainingContext] = {}
        self._models: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Audit & Availability Matrix
    # ------------------------------------------------------------------
    def _load_training_records(self) -> Dict[str, Dict[str, Any]]:
        tr_path = self.metrics_dir / "training_results.json"
        if not tr_path.exists():
            return {}
        records = json.loads(tr_path.read_text(encoding="utf-8"))
        return {r["exp_id"]: r for r in records}

    def _load_manifest_types(self) -> Dict[str, str]:
        man_path = self.metrics_dir / "checkpoint_integrity_manifest.csv"
        if not man_path.exists():
            return {}
        df = pd.read_csv(man_path)
        return dict(zip(df["experiment_id"], df["type"]))

    def _build_availability_matrix(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for scale in SUPPORTED_SCALES:
            for district in SUPPORTED_DISTRICTS:
                feat_parquet = self.data_dir / f"{district}_features.parquet"
                ind_parquet = self.data_dir / f"{district}_individual_meters.parquet"
                for arch in SUPPORTED_ARCHITECTURES:
                    for f_cfg in SUPPORTED_FEATURE_CONFIGS:
                        exp_id = make_experiment_id(scale, district, arch, f_cfg)
                        ckpt_path = self.checkpoints_dir / f"{exp_id}.pt"
                        rec = self._training_records.get(exp_id, {})
                        man_type = str(self._manifest_types.get(exp_id, ""))

                        supported = False
                        status = "unavailable"
                        reason = ""
                        ckpt_epoch = None
                        logged_epoch = rec.get("best_epoch")

                        if scale == "individual":
                            if ckpt_path.exists() and "partial" in man_type:
                                status = "partial_checkpoint_excluded"
                                reason = (
                                    "Local checkpoint is an interrupted partial run (epoch 3, val_loss 0.1244 "
                                    f"vs finalized Colab epoch {logged_epoch}, val_loss {rec.get('best_val_loss', 0.1214):.4f}); "
                                    "excluded to prevent inaccurate inference."
                                )
                            elif not ckpt_path.exists():
                                status = "checkpoint_pending_colab_export"
                                reason = (
                                    "Model was trained in Google Colab (Phase 3.1), but finalized .pt weights "
                                    "and per-meter scalers are pending Phase 3.3 export. Aggregate models are "
                                    "never substituted for individual-meter inference."
                                )
                            else:
                                status = "unsupported_scale"
                                reason = "Individual-meter simulator requires exported finalized per-meter artifacts."
                        else:
                            if not TORCH_AVAILABLE:
                                status = "missing_pytorch"
                                reason = "PyTorch runtime is not installed in the active Python environment."
                            elif not feat_parquet.exists():
                                status = "missing_preprocessing_data"
                                reason = f"Processed training dataset {feat_parquet.name} is missing."
                            elif not ckpt_path.exists():
                                status = "missing_checkpoint"
                                reason = f"Trained checkpoint {ckpt_path.name} is not present locally."
                            elif "partial" in man_type:
                                status = "partial_checkpoint_excluded"
                                reason = f"Checkpoint {ckpt_path.name} is marked partial in integrity manifest."
                            else:
                                supported = True
                                status = "supported"
                                ckpt_epoch = logged_epoch
                                reason = (
                                    "Verified finalized PyTorch checkpoint and deterministic 70%-train "
                                    "StandardScaler artifacts available locally."
                                )

                        rows.append({
                            "exp_id": exp_id,
                            "scale": scale,
                            "district": district,
                            "arch": arch,
                            "feature_config": f_cfg,
                            "in_features": len(FEATURE_SETS[f_cfg]),
                            "feature_columns": list(FEATURE_SETS[f_cfg]),
                            "supported": supported,
                            "status": status,
                            "reason": reason,
                            "checkpoint_file": ckpt_path.name if ckpt_path.exists() else None,
                            "checkpoint_sha256_16": _sha256_short(ckpt_path) if ckpt_path.exists() else None,
                            "best_epoch": logged_epoch if supported else ckpt_epoch,
                            "test_rmse_kw": (rec.get("metrics") or {}).get("rmse_kw"),
                            "test_r2": (rec.get("metrics") or {}).get("r2_score"),
                            "has_data_parquet": feat_parquet.exists() if scale == "aggregate" else ind_parquet.exists(),
                        })
        return rows

    def get_availability_matrix(self) -> List[Dict[str, Any]]:
        return list(self._availability)

    def check_combination(
        self, scale: str, district: str, arch: str, feature_config: str
    ) -> Dict[str, Any]:
        if scale not in SUPPORTED_SCALES:
            raise UnsupportedConfigurationError(
                f"Unsupported scale '{scale}'. Supported scales: {list(SUPPORTED_SCALES)}."
            )
        if district not in SUPPORTED_DISTRICTS:
            raise UnsupportedConfigurationError(
                f"Unsupported district '{district}'. Supported districts: {list(SUPPORTED_DISTRICTS)}."
            )
        if arch not in SUPPORTED_ARCHITECTURES:
            raise UnsupportedConfigurationError(
                f"Unsupported architecture '{arch}'. Supported architectures: {list(SUPPORTED_ARCHITECTURES)}."
            )
        if feature_config not in SUPPORTED_FEATURE_CONFIGS:
            raise UnsupportedConfigurationError(
                f"Unsupported feature configuration '{feature_config}'. Supported: {list(SUPPORTED_FEATURE_CONFIGS)}."
            )
        exp_id = make_experiment_id(scale, district, arch, feature_config)
        info = self._availability_by_id[exp_id]
        if not info["supported"]:
            raise UnsupportedConfigurationError(f"[{exp_id}] Unsupported for inference: {info['reason']}")
        return info

    # ------------------------------------------------------------------
    # 2. Exact Training Context, Scalers & Regime Pools
    # ------------------------------------------------------------------
    def _get_context(self, district: str) -> DistrictTrainingContext:
        if district not in SUPPORTED_DISTRICTS:
            raise UnsupportedConfigurationError(f"Unsupported district '{district}'.")
        if district in self._contexts:
            return self._contexts[district]

        feat_path = self.data_dir / f"{district}_features.parquet"
        if not feat_path.exists():
            raise UnsupportedConfigurationError(f"Missing processed feature file: {feat_path}")

        df_feat = pd.read_parquet(feat_path)
        target_col = "aggregate_demand_kw_primary"

        X_train_by_cfg: Dict[str, np.ndarray] = {}
        y_train_by_cfg: Dict[str, np.ndarray] = {}
        ts_train_by_cfg: Dict[str, np.ndarray] = {}
        X_test_by_cfg: Dict[str, np.ndarray] = {}
        y_test_by_cfg: Dict[str, np.ndarray] = {}
        ts_test_by_cfg: Dict[str, np.ndarray] = {}
        feat_scalers: Dict[str, StandardScaler] = {}
        targ_scalers: Dict[str, StandardScaler] = {}
        feature_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

        for f_name in SUPPORTED_FEATURE_CONFIGS:
            f_cols = FEATURE_SETS[f_name]
            X, y, ts = extract_contiguous_sequences(
                df_feat, target_col, f_cols, lookback_len=LOOKBACK_STEPS, horizon_len=HORIZON_STEPS
            )
            (X_tr, y_tr, ts_tr), _, (X_te, y_te, ts_te) = temporal_train_val_test_split(
                X, y, ts, train_ratio=0.70, val_ratio=0.15, purge_gap=LOOKBACK_STEPS
            )

            n_feats = len(f_cols)
            f_scaler = StandardScaler()
            t_scaler = StandardScaler()
            f_scaler.fit(X_tr.reshape(-1, n_feats))
            t_scaler.fit(y_tr)

            X_train_by_cfg[f_name] = X_tr
            y_train_by_cfg[f_name] = y_tr
            ts_train_by_cfg[f_name] = ts_tr
            X_test_by_cfg[f_name] = X_te
            y_test_by_cfg[f_name] = y_te
            ts_test_by_cfg[f_name] = ts_te
            feat_scalers[f_name] = f_scaler
            targ_scalers[f_name] = t_scaler

            stats_for_cfg: Dict[str, Dict[str, float]] = {}
            flat = X_tr.reshape(-1, n_feats)
            for idx, col in enumerate(f_cols):
                col_vals = flat[:, idx]
                qs = np.quantile(col_vals, [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0])
                stats_for_cfg[col] = {
                    "min": float(qs[0]),
                    "p01": float(qs[1]),
                    "p05": float(qs[2]),
                    "p25": float(qs[3]),
                    "p50": float(qs[4]),
                    "p75": float(qs[5]),
                    "p95": float(qs[6]),
                    "p99": float(qs[7]),
                    "max": float(qs[8]),
                    "mean": float(np.mean(col_vals)),
                    "std": float(np.std(col_vals)),
                }
            feature_stats[f_name] = stats_for_cfg

        # Build empirical scenario regime pools from consumption_only training windows
        # (Note: all 3 configs share the exact same 6,303 Mathura and 6,009 Bareilly window indices!)
        c_tr = X_train_by_cfg["consumption_only"][:, :, 0]  # (N_train, 16)
        ts_tr_dt = pd.to_datetime(ts_train_by_cfg["consumption_only"])
        end_dt = ts_tr_dt - pd.Timedelta(minutes=INTERVAL_MINUTES)
        end_hr = end_dt.hour + end_dt.minute / 60.0

        w_mean = c_tr.mean(axis=1)
        w_max = c_tr.max(axis=1)
        w_diff = np.abs(np.diff(c_tr, axis=1)).mean(axis=1)

        morn_window = (end_hr >= 6.5) & (end_hr <= 10.75)
        morn_med = float(np.median(w_max[morn_window])) if np.any(morn_window) else float(np.median(w_max))

        even_window = (end_hr >= 18.5) & (end_hr <= 22.75)
        even_med = float(np.median(w_max[even_window])) if np.any(even_window) else float(np.median(w_max))

        masks = {
            "normal": (
                (w_mean >= np.quantile(w_mean, 0.35))
                & (w_mean <= np.quantile(w_mean, 0.65))
                & (w_diff <= np.quantile(w_diff, 0.75))
            ),
            "high_demand": w_mean >= np.quantile(w_mean, 0.90),
            "low_demand": w_mean <= np.quantile(w_mean, 0.10),
            "morning_peak": morn_window & (w_max >= morn_med),
            "evening_peak": even_window & (w_max >= even_med) & ((c_tr[:, -1] - c_tr[:, 0]) > 0),
            "irregular": w_diff >= np.quantile(w_diff, 0.90),
        }

        regime_indices: Dict[str, np.ndarray] = {}
        regime_chol: Dict[str, np.ndarray] = {}
        for key, mask in masks.items():
            idx = np.flatnonzero(mask)
            if len(idx) < 10:
                idx = np.arange(len(c_tr))
            regime_indices[key] = idx
            pool = c_tr[idx]
            centered = pool - pool.mean(axis=0, keepdims=True)
            cov = (centered.T @ centered) / max(1, len(pool) - 1)
            # Regularize diagonal slightly for positive-definite Cholesky factor
            cov_reg = cov + np.eye(LOOKBACK_STEPS) * 1e-4
            regime_chol[key] = np.linalg.cholesky(cov_reg)

        ctx = DistrictTrainingContext(
            district=district,
            X_train_by_config=X_train_by_cfg,
            y_train_by_config=y_train_by_cfg,
            ts_train_by_config=ts_train_by_cfg,
            X_test_by_config=X_test_by_cfg,
            y_test_by_config=y_test_by_cfg,
            ts_test_by_config=ts_test_by_cfg,
            feat_scalers=feat_scalers,
            targ_scalers=targ_scalers,
            feature_stats=feature_stats,
            regime_indices=regime_indices,
            regime_chol=regime_chol,
        )
        self._contexts[district] = ctx
        return ctx

    def _get_model(self, exp_id: str, arch: str, in_features: int):
        if exp_id in self._models:
            return self._models[exp_id]
        if exp_id not in self._availability_by_id or not self._availability_by_id[exp_id]["supported"]:
            raise UnsupportedConfigurationError(f"Model '{exp_id}' is not in the supported allowlist.")
        ckpt_path = (self.checkpoints_dir / f"{exp_id}.pt").resolve()
        if not ckpt_path.is_relative_to(self.checkpoints_dir.resolve()):
            raise UnsupportedConfigurationError("Invalid checkpoint path resolution.")
        chk = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in chk:
            raise UnsupportedConfigurationError(f"Checkpoint {ckpt_path.name} missing 'model_state_dict'.")
        model = get_model(arch, in_features, sequence_length=LOOKBACK_STEPS)
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        self._models[exp_id] = (model, chk)
        return model, chk

    # ------------------------------------------------------------------
    # 3. Validation & Out-of-Distribution (OOD) Analysis
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_finite_float(val: Any, label: str) -> float:
        if val is None or isinstance(val, bool):
            raise SimulatorValidationError(f"Missing or boolean value at {label}.")
        if isinstance(val, str):
            val = val.strip()
            if val == "":
                raise SimulatorValidationError(f"Missing empty value at {label}.")
        try:
            f = float(val)
        except (TypeError, ValueError) as exc:
            raise SimulatorValidationError(f"Non-numeric value '{val}' at {label}.") from exc
        if math.isnan(f) or math.isinf(f):
            raise SimulatorValidationError(f"NaN or infinite value at {label}.")
        return f

    def validate_inputs(
        self,
        district: str,
        feature_config: str,
        consumption_kw: Sequence[Any],
        weather_rows: Optional[Sequence[Sequence[Any]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validate 16 consecutive consumption observations (and 16x9 weather features if required).

        Returns:
            window_matrix: float64 array of shape (16, in_features) in exact FEATURE_SETS order.
                           Unusual / OOD values are NEVER silently clipped.
            report: structured validation & OOD diagnostic dictionary.
        """
        ctx = self._get_context(district)
        if feature_config not in SUPPORTED_FEATURE_CONFIGS:
            raise UnsupportedConfigurationError(f"Unsupported feature_config '{feature_config}'.")

        if consumption_kw is None or isinstance(consumption_kw, (str, bytes)):
            raise SimulatorValidationError("Consumption input must be a sequence of 16 numeric kW values.")
        cons_list = list(consumption_kw)
        if len(cons_list) != LOOKBACK_STEPS:
            raise SimulatorValidationError(
                f"Expected exactly {LOOKBACK_STEPS} consecutive consumption intervals (4 hours), "
                f"but received {len(cons_list)}."
            )

        clean_cons = np.empty(LOOKBACK_STEPS, dtype=np.float64)
        for i, raw in enumerate(cons_list):
            v = self._coerce_finite_float(raw, f"consumption step t-{LOOKBACK_STEPS - 1 - i} (index {i + 1})")
            if v < 0.0:
                raise SimulatorValidationError(
                    f"Invalid negative electricity demand ({v} kW) at step {i + 1}; demand must be >= 0 kW."
                )
            clean_cons[i] = v

        f_cols = FEATURE_SETS[feature_config]
        n_feats = len(f_cols)
        W = np.zeros((LOOKBACK_STEPS, n_feats), dtype=np.float64)
        W[:, 0] = clean_cons

        wx_cols = f_cols[1:]
        if wx_cols:
            if weather_rows is None:
                raise SimulatorValidationError(
                    f"Feature configuration '{feature_config}' requires exact 16-step weather inputs "
                    f"for all {len(wx_cols)} trained meteorological variables: {wx_cols}. "
                    "Weather values are never fabricated."
                )
            wx_list = list(weather_rows)
            if len(wx_list) != LOOKBACK_STEPS:
                raise SimulatorValidationError(
                    f"Weather input must have {LOOKBACK_STEPS} consecutive rows, received {len(wx_list)}."
                )
            for r_idx, row in enumerate(wx_list):
                if row is None or isinstance(row, (str, bytes)) or len(row) != len(wx_cols):
                    got = 0 if row is None or isinstance(row, (str, bytes)) else len(row)
                    raise SimulatorValidationError(
                        f"Weather row {r_idx + 1} must contain {len(wx_cols)} values ({wx_cols}), received {got}."
                    )
                for c_idx, col_name in enumerate(wx_cols):
                    val = self._coerce_finite_float(row[c_idx], f"{col_name} at step {r_idx + 1}")
                    phys_err = _weather_physical_bounds_error(col_name, val)
                    if phys_err:
                        raise SimulatorValidationError(f"Step {r_idx + 1}: {phys_err}.")
                    W[r_idx, c_idx + 1] = val

        # Distribution range check against district training split (never clips!)
        stats_cfg = ctx.feature_stats[feature_config]
        warnings: List[Dict[str, Any]] = []
        c_stats = stats_cfg["demand_target"]

        below_min = [i + 1 for i, v in enumerate(clean_cons) if v < c_stats["min"]]
        above_max = [i + 1 for i, v in enumerate(clean_cons) if v > c_stats["max"]]
        tail_low = [i + 1 for i, v in enumerate(clean_cons) if c_stats["min"] <= v < c_stats["p01"]]
        tail_high = [i + 1 for i, v in enumerate(clean_cons) if c_stats["p99"] < v <= c_stats["max"]]

        if below_min or above_max:
            warnings.append({
                "feature": "demand_target",
                "severity": "high",
                "code": "consumption_outside_training_range",
                "message": (
                    f"Consumption values outside {district.capitalize()} training range "
                    f"[{c_stats['min']:.3f}, {c_stats['max']:.3f}] kW at step(s) "
                    f"{below_min + above_max}. Values were preserved without clipping."
                ),
                "steps_below_min": below_min,
                "steps_above_max": above_max,
            })
        elif tail_low or tail_high:
            warnings.append({
                "feature": "demand_target",
                "severity": "moderate",
                "code": "consumption_in_extreme_tail",
                "message": (
                    f"Consumption enters {district.capitalize()} 1%–99% training tail "
                    f"([p01={c_stats['p01']:.3f}, p99={c_stats['p99']:.3f}] kW) at step(s) "
                    f"{tail_low + tail_high}. Values were preserved without clipping."
                ),
                "steps_tail_low": tail_low,
                "steps_tail_high": tail_high,
            })

        for c_idx, col_name in enumerate(wx_cols):
            st = stats_cfg[col_name]
            col_vec = W[:, c_idx + 1]
            out_steps = [i + 1 for i, v in enumerate(col_vec) if v < st["min"] or v > st["max"]]
            if out_steps:
                warnings.append({
                    "feature": col_name,
                    "severity": "high",
                    "code": "weather_outside_training_range",
                    "message": (
                        f"{col_name} outside {district.capitalize()} training range "
                        f"[{st['min']:.2f}, {st['max']:.2f}] at step(s) {out_steps}. "
                        "Values preserved without clipping."
                    ),
                    "steps": out_steps,
                })

        status = "valid_with_ood_warning" if warnings else "valid_in_distribution"
        report = {
            "valid": True,
            "status": status,
            "clipped": False,
            "lookback_steps": LOOKBACK_STEPS,
            "in_features": n_feats,
            "feature_order": list(f_cols),
            "warnings": warnings,
            "training_consumption_range_kw": c_stats,
            "input_summary_kw": {
                "min": float(np.min(clean_cons)),
                "max": float(np.max(clean_cons)),
                "mean": float(np.mean(clean_cons)),
                "std": float(np.std(clean_cons)),
                "last_t0": float(clean_cons[-1]),
            },
        }
        return W, report

    # ------------------------------------------------------------------
    # 4. Statistically Grounded Random Scenario Generator
    # ------------------------------------------------------------------
    def generate_scenario(
        self,
        district: str,
        scenario: str = "normal",
        seed: Optional[int] = 42,
        perturb_scale: float = 0.12,
    ) -> Dict[str, Any]:
        """
        Generate 16 consecutive consumption observations (in kW) grounded in the
        selected district's 70% training split, preserving temporal autocorrelation
        and diurnal shapes. Also returns the paired real training weather matrices
        from the sampled historical window for optional pipeline-compatible weather use.
        """
        ctx = self._get_context(district)
        if scenario not in SCENARIO_PRESETS:
            raise SimulatorValidationError(
                f"Unknown scenario '{scenario}'. Valid scenarios: {list(SCENARIO_PRESETS)}."
            )

        rng = np.random.default_rng(seed)
        pool_idx = ctx.regime_indices[scenario]
        chosen_pos = int(rng.integers(0, len(pool_idx)))
        win_idx = int(pool_idx[chosen_pos])

        base_cons = ctx.X_train_by_config["consumption_only"][win_idx, :, 0].astype(np.float64)
        c_stats = ctx.feature_stats["consumption_only"]["demand_target"]

        if perturb_scale > 0.0:
            z = rng.standard_normal(LOOKBACK_STEPS)
            # Correlated innovation preserving the empirical 16x16 lag covariance of the regime
            delta = float(perturb_scale) * (ctx.regime_chol[scenario] @ z)
            floor_kw = max(0.05, c_stats["min"] * 0.5)
            cons = np.maximum(floor_kw, base_cons + delta)
        else:
            cons = base_cons.copy()

        target_ts = pd.Timestamp(ctx.ts_train_by_config["consumption_only"][win_idx])
        start_ts = target_ts - pd.Timedelta(minutes=INTERVAL_MINUTES * LOOKBACK_STEPS)
        step_timestamps = [
            (start_ts + pd.Timedelta(minutes=INTERVAL_MINUTES * k)).strftime("%Y-%m-%dT%H:%M")
            for k in range(LOOKBACK_STEPS)
        ]

        wx_retro = ctx.X_train_by_config["consumption_weather_retro"][win_idx, :, 1:]
        wx_oper = ctx.X_train_by_config["consumption_weather_operational"][win_idx, :, 1:]

        return {
            "district": district,
            "scenario": scenario,
            "scenario_label": SCENARIO_PRESETS[scenario]["label"],
            "scenario_description": SCENARIO_PRESETS[scenario]["description"],
            "seed": seed,
            "source_train_window_index": win_idx,
            "regime_pool_size": int(len(pool_idx)),
            "reference_window_timestamps": step_timestamps,
            "consumption_kw": [round(float(x), 4) for x in cons],
            "base_training_consumption_kw": [round(float(x), 4) for x in base_cons],
            "paired_weather": {
                "consumption_weather_retro": {
                    "columns": weather_feature_names("consumption_weather_retro"),
                    "rows": [[round(float(v), 4) for v in row] for row in wx_retro],
                },
                "consumption_weather_operational": {
                    "columns": weather_feature_names("consumption_weather_operational"),
                    "rows": [[round(float(v), 4) for v in row] for row in wx_oper],
                },
            },
        }

    # ------------------------------------------------------------------
    # 5. Pipeline-Compatible Historical Weather Window Retrieval
    # ------------------------------------------------------------------
    def list_historical_weather_windows(
        self, district: str, split: str = "test", max_windows: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Return representative real 16-step historical weather windows from the
        district's processed dataset (compatible with the training pipeline).
        """
        ctx = self._get_context(district)
        if split == "train":
            X_ret = ctx.X_train_by_config["consumption_weather_retro"]
            X_ope = ctx.X_train_by_config["consumption_weather_operational"]
            ts_arr = ctx.ts_train_by_config["consumption_only"]
        else:
            X_ret = ctx.X_test_by_config["consumption_weather_retro"]
            X_ope = ctx.X_test_by_config["consumption_weather_operational"]
            ts_arr = ctx.ts_test_by_config["consumption_only"]

        n_total = len(ts_arr)
        indices = np.linspace(0, n_total - 1, num=min(max_windows, n_total), dtype=int)
        out: List[Dict[str, Any]] = []
        for idx in indices:
            t_end = pd.Timestamp(ts_arr[idx]) - pd.Timedelta(minutes=INTERVAL_MINUTES)
            t_start = t_end - pd.Timedelta(minutes=INTERVAL_MINUTES * (LOOKBACK_STEPS - 1))
            temp_mean = float(np.mean(X_ret[idx, :, 1]))
            rh_mean = float(np.mean(X_ret[idx, :, 2]))
            out.append({
                "window_index": int(idx),
                "split": split,
                "start_timestamp": t_start.strftime("%Y-%m-%d %H:%M"),
                "end_timestamp": t_end.strftime("%Y-%m-%d %H:%M"),
                "target_timestamp": pd.Timestamp(ts_arr[idx]).strftime("%Y-%m-%d %H:%M"),
                "summary": f"{t_end.strftime('%Y-%m-%d %H:%M')} · {temp_mean:.1f}°C, {rh_mean:.0f}% RH",
                "consumption_kw": [round(float(v), 4) for v in X_ret[idx, :, 0]],
                "consumption_weather_retro": [[round(float(v), 4) for v in r] for r in X_ret[idx, :, 1:]],
                "consumption_weather_operational": [[round(float(v), 4) for v in r] for r in X_ope[idx, :, 1:]],
            })
        return out

    # ------------------------------------------------------------------
    # 6. CSV Parsing
    # ------------------------------------------------------------------
    def parse_csv_input(
        self,
        csv_text: str,
        feature_config: str = "consumption_only",
        row_offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Parse uploaded CSV text into 16 consecutive consumption values (and 16x9
        weather rows if `feature_config` requires weather).
        """
        if not csv_text or not csv_text.strip():
            raise SimulatorValidationError("Uploaded CSV is empty.")

        reader = csv.reader(io.StringIO(csv_text.strip()))
        raw_rows = [r for r in reader if any(cell.strip() != "" for cell in r)]
        if not raw_rows:
            raise SimulatorValidationError("Uploaded CSV contains no non-empty rows.")

        # Detect whether row 0 is a header
        first = [c.strip() for c in raw_rows[0]]
        has_header = False
        try:
            for c in first:
                float(c)
        except ValueError:
            has_header = True

        wx_cols = weather_feature_names(feature_config)
        if not has_header:
            if wx_cols:
                raise SimulatorValidationError(
                    f"Weather-dependent configuration '{feature_config}' requires a CSV header row "
                    f"naming consumption and all 9 weather columns: ['demand_target'] + {wx_cols}."
                )
            if all(len(r) == 1 for r in raw_rows):
                data_rows = raw_rows
                headers = ["demand_target"]
            elif len(raw_rows) == 1 and len(raw_rows[0]) == LOOKBACK_STEPS:
                data_rows = [[v] for v in raw_rows[0]]
                headers = ["demand_target"]
            else:
                raise SimulatorValidationError(
                    "Headerless CSV must contain either 1 column of 16 rows or 1 row of 16 values, "
                    "or include a header row."
                )
        else:
            headers = [h.strip() for h in raw_rows[0]]
            data_rows = raw_rows[1:]

        total_rows = len(data_rows)
        if total_rows < LOOKBACK_STEPS:
            raise SimulatorValidationError(
                f"CSV must contain at least {LOOKBACK_STEPS} consecutive data rows; found {total_rows}."
            )

        if row_offset < 0:
            start = total_rows - LOOKBACK_STEPS
        else:
            start = int(row_offset)
        if start < 0 or start + LOOKBACK_STEPS > total_rows:
            raise SimulatorValidationError(
                f"CSV row_offset={row_offset} out of range for {total_rows} data rows "
                f"(valid start range: 0..{total_rows - LOOKBACK_STEPS})."
            )
        selected = data_rows[start : start + LOOKBACK_STEPS]

        lower_map = {h.lower(): idx for idx, h in enumerate(headers)}
        cons_col_idx: Optional[int] = None
        for alias in CONSUMPTION_COLUMN_ALIASES:
            if alias.lower() in lower_map:
                cons_col_idx = lower_map[alias.lower()]
                break
        if cons_col_idx is None:
            if len(headers) == 1 and not wx_cols:
                cons_col_idx = 0
            else:
                raise SimulatorValidationError(
                    f"CSV is missing a recognized consumption column (accepted names: {list(CONSUMPTION_COLUMN_ALIASES)}). "
                    f"Found columns: {headers}."
                )

        # Optional timestamp contiguity verification if a `timestamp` column is present
        timestamps: Optional[List[str]] = None
        if "timestamp" in lower_map:
            ts_idx = lower_map["timestamp"]
            ts_raw = [r[ts_idx].strip() for r in selected if len(r) > ts_idx]
            try:
                parsed_ts = pd.to_datetime(ts_raw)
                if parsed_ts.duplicated().any():
                    raise SimulatorValidationError(
                        "CSV `timestamp` column contains duplicate timestamps across the selected 16 rows."
                    )
                diffs = parsed_ts.to_series().diff().dropna()
                expected = pd.Timedelta(minutes=INTERVAL_MINUTES)
                if not (diffs == expected).all():
                    raise SimulatorValidationError(
                        "CSV `timestamp` column is not strictly consecutive at 15-minute intervals "
                        "across the selected 16 rows."
                    )
                timestamps = [t.strftime("%Y-%m-%d %H:%M") for t in parsed_ts]
            except SimulatorValidationError:
                raise
            except Exception as exc:
                raise SimulatorValidationError(f"Invalid timestamp in CSV: {exc}") from exc

        cons_vals: List[float] = []
        for i, r in enumerate(selected):
            if len(r) <= cons_col_idx:
                raise SimulatorValidationError(f"CSV row {start + i + 1} is missing consumption column.")
            cons_vals.append(self._coerce_finite_float(r[cons_col_idx], f"CSV row {start + i + 1} consumption"))

        wx_matrix: Optional[List[List[float]]] = None
        if wx_cols:
            missing_wx = [c for c in wx_cols if c.lower() not in lower_map]
            if missing_wx:
                raise SimulatorValidationError(
                    f"CSV is missing required trained weather column(s) for '{feature_config}': {missing_wx}."
                )
            wx_indices = [lower_map[c.lower()] for c in wx_cols]
            wx_matrix = []
            for i, r in enumerate(selected):
                wx_row: List[float] = []
                for col_name, col_i in zip(wx_cols, wx_indices):
                    if len(r) <= col_i:
                        raise SimulatorValidationError(f"CSV row {start + i + 1} missing column '{col_name}'.")
                    wx_row.append(self._coerce_finite_float(r[col_i], f"CSV row {start + i + 1} {col_name}"))
                wx_matrix.append(wx_row)

        return {
            "total_rows": total_rows,
            "row_offset": start,
            "headers": headers,
            "timestamps": timestamps,
            "consumption_kw": cons_vals,
            "weather_rows": wx_matrix,
        }

    # ------------------------------------------------------------------
    # 7. Exact Trained Model Prediction Pipeline
    # ------------------------------------------------------------------
    def predict_next_interval(
        self,
        scale: str,
        district: str,
        arch: str,
        feature_config: str,
        consumption_kw: Sequence[Any],
        weather_rows: Optional[Sequence[Sequence[Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute next-15-minute demand prediction using the verified PyTorch checkpoint
        and matching training-split StandardScalers.
        """
        avail = self.check_combination(scale, district, arch, feature_config)
        exp_id = avail["exp_id"]
        ctx = self._get_context(district)

        W_raw, val_report = self.validate_inputs(
            district=district,
            feature_config=feature_config,
            consumption_kw=consumption_kw,
            weather_rows=weather_rows,
        )

        feat_scaler = ctx.feat_scalers[feature_config]
        targ_scaler = ctx.targ_scalers[feature_config]
        n_feats = len(FEATURE_SETS[feature_config])

        # Exact identical scaling transformation as src/train_runner.py
        W_scaled = feat_scaler.transform(W_raw).reshape(1, LOOKBACK_STEPS, n_feats).astype(np.float32)

        model, chk = self._get_model(exp_id, arch, n_feats)
        with torch.no_grad():
            pred_scaled = model(torch.from_numpy(W_scaled)).cpu().numpy()

        pred_kw = float(targ_scaler.inverse_transform(pred_scaled)[0, 0])
        persistence_kw = float(W_raw[-1, 0])
        delta_kw = pred_kw - persistence_kw
        delta_pct = (delta_kw / persistence_kw * 100.0) if persistence_kw > 1e-9 else None

        return {
            "exp_id": exp_id,
            "engine": "pytorch_eval",
            "scale": scale,
            "district": district,
            "architecture": arch,
            "feature_config": feature_config,
            "horizon_minutes": INTERVAL_MINUTES,
            "lookback_intervals": LOOKBACK_STEPS,
            "predicted_demand_kw": pred_kw,
            "naive_persistence_kw": persistence_kw,
            "delta_vs_persistence_kw": delta_kw,
            "delta_vs_persistence_pct": delta_pct,
            "historical_consumption_kw": [float(x) for x in W_raw[:, 0]],
            "validation": val_report,
            "checkpoint": {
                "file": avail["checkpoint_file"],
                "sha256_16": avail["checkpoint_sha256_16"],
                "best_epoch": int(chk.get("epoch", avail["best_epoch"] or 0)),
                "val_loss": float(chk.get("val_loss", 0.0)),
                "test_rmse_kw": avail["test_rmse_kw"],
                "test_r2": avail["test_r2"],
            },
        }

