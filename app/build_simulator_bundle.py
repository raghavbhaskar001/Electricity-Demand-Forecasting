"""
Compiler for `app/dashboard/simulator_bundle.js`.

Exports verified training scalers, feature distribution bounds, the 36-cell
availability matrix, statistically grounded training-regime pools + empirical
16x16 lag-covariance Cholesky factors, pipeline-compatible historical weather
windows, and exact IEEE-754 float32 weights from all 18 finalized aggregate
PyTorch checkpoints (`artifacts/checkpoints/agg_*.pt`).

This enables `app/dashboard/index.html` and `app/app.py` (Streamlit) to run
100% genuine checkpoint inference in evaluation mode even without an active
HTTP API server, while `app/serve_dashboard.py` also serves live Python/PyTorch
`/api/predict` and `/api/scenario` endpoints.

Usage:
    python app/build_simulator_bundle.py
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import FEATURE_SETS  # noqa: E402
from src.simulator import (  # noqa: E402
    INTERVAL_MINUTES,
    LOOKBACK_STEPS,
    SCENARIO_PRESETS,
    SUPPORTED_DISTRICTS,
    SUPPORTED_FEATURE_CONFIGS,
    WEATHER_UNITS,
    PredictionSimulator,
    weather_feature_names,
)

OUT = ROOT / "app" / "dashboard" / "simulator_bundle.js"


def encode_f32_tensor(arr: np.ndarray) -> Dict[str, Any]:
    a = np.ascontiguousarray(arr, dtype="<f4")
    return {
        "shape": list(a.shape),
        "b64": base64.b64encode(a.tobytes()).decode("ascii"),
    }


def build_bundle() -> Dict[str, Any]:
    sim = PredictionSimulator(ROOT)
    availability = sim.get_availability_matrix()

    districts_payload: Dict[str, Any] = {}
    for district in SUPPORTED_DISTRICTS:
        ctx = sim._get_context(district)

        scalers_out: Dict[str, Any] = {}
        stats_out: Dict[str, Any] = {}
        for f_cfg in SUPPORTED_FEATURE_CONFIGS:
            fs = ctx.feat_scalers[f_cfg]
            ts = ctx.targ_scalers[f_cfg]
            scalers_out[f_cfg] = {
                "feature_columns": list(FEATURE_SETS[f_cfg]),
                "feat_mean": [float(x) for x in fs.mean_.ravel()],
                "feat_scale": [float(x) for x in fs.scale_.ravel()],
                "targ_mean": float(ts.mean_.ravel()[0]),
                "targ_scale": float(ts.scale_.ravel()[0]),
            }
            stats_out[f_cfg] = ctx.feature_stats[f_cfg]

        # Stratified real training windows across all 6 scenario regimes (up to 45 per regime)
        selected_set = set()
        regime_full_indices: Dict[str, List[int]] = {}
        for s_key, idx_arr in ctx.regime_indices.items():
            k = min(48, len(idx_arr))
            pos = np.linspace(0, len(idx_arr) - 1, num=k, dtype=int)
            picked = [int(idx_arr[p]) for p in pos]
            regime_full_indices[s_key] = picked
            selected_set.update(picked)

        sorted_win_ids = sorted(selected_set)
        win_id_to_local = {wid: loc for loc, wid in enumerate(sorted_win_ids)}

        X_ret_tr = ctx.X_train_by_config["consumption_weather_retro"]
        X_ope_tr = ctx.X_train_by_config["consumption_weather_operational"]
        ts_tr = ctx.ts_train_by_config["consumption_only"]

        train_pool: List[Dict[str, Any]] = []
        for wid in sorted_win_ids:
            t_end = pd.Timestamp(ts_tr[wid]) - pd.Timedelta(minutes=INTERVAL_MINUTES)
            train_pool.append({
                "train_idx": wid,
                "end_ts": t_end.strftime("%Y-%m-%d %H:%M"),
                "c": [round(float(v), 4) for v in X_ret_tr[wid, :, 0]],
                "retro": [[round(float(v), 3) for v in r] for r in X_ret_tr[wid, :, 1:]],
                "oper": [[round(float(v), 3) for v in r] for r in X_ope_tr[wid, :, 1:]],
            })

        regimes_out: Dict[str, Any] = {}
        for s_key in SCENARIO_PRESETS:
            regimes_out[s_key] = {
                "full_pool_size": int(len(ctx.regime_indices[s_key])),
                "pool_refs": [win_id_to_local[wid] for wid in regime_full_indices[s_key]],
                "chol": [[round(float(v), 6) for v in row] for row in ctx.regime_chol[s_key]],
            }

        districts_payload[district] = {
            "scalers": scalers_out,
            "feature_stats": stats_out,
            "train_pool": train_pool,
            "regimes": regimes_out,
            "historical_weather_windows": sim.list_historical_weather_windows(
                district, split="test", max_windows=24
            ),
        }

    # Export float32 weights for all 18 supported aggregate checkpoints
    models_payload: Dict[str, Any] = {}
    for row in availability:
        if not row["supported"]:
            continue
        exp_id = row["exp_id"]
        ckpt_path = ROOT / "artifacts" / "checkpoints" / f"{exp_id}.pt"
        chk = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = chk["model_state_dict"]
        tensors: Dict[str, Any] = {}
        for k, tensor in sd.items():
            if k.endswith("num_batches_tracked"):
                continue
            tensors[k] = encode_f32_tensor(tensor.detach().cpu().numpy())
        models_payload[exp_id] = {
            "arch": row["arch"],
            "in_features": row["in_features"],
            "epoch": int(chk.get("epoch", row["best_epoch"] or 0)),
            "val_loss": float(chk.get("val_loss", 0.0)),
            "sha256_16": row["checkpoint_sha256_16"],
            "tensors": tensors,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lookback_steps": LOOKBACK_STEPS,
        "horizon_minutes": INTERVAL_MINUTES,
        "feature_sets": FEATURE_SETS,
        "weather_columns": {
            "consumption_weather_retro": weather_feature_names("consumption_weather_retro"),
            "consumption_weather_operational": weather_feature_names("consumption_weather_operational"),
        },
        "weather_units": WEATHER_UNITS,
        "scenario_presets": SCENARIO_PRESETS,
        "availability": availability,
        "districts": districts_payload,
        "models": models_payload,
    }


def main() -> None:
    payload = build_bundle()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "// AUTO-GENERATED by app/build_simulator_bundle.py — do not edit by hand.\n"
        "window.SIMULATOR_BUNDLE = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    n_sup = sum(1 for r in payload["availability"] if r["supported"])
    print(
        f"Simulator bundle ready: {n_sup}/{len(payload['availability'])} supported models | "
        f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()

