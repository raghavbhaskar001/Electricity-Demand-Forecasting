"""
Dashboard data builder for the Midnight Grid research dashboard.

Reads ONLY verified project artifacts and writes `app/dashboard/data.js`
(`window.DASHBOARD_DATA = {...}`) so the static dashboard can be opened
directly from disk without a server or network access.

Sources (read-only):
  artifacts/metrics/training_results.json            -> 36 experiment records + test metrics
  artifacts/metrics/phase_3_4_consolidated_results.csv -> statistical validation status
  artifacts/metrics/baseline_benchmarks.json         -> persistence baselines + sequence census
  artifacts/metrics/aggregate_test_predictions.parquet -> pointwise aggregate test predictions
  artifacts/metrics/dm_test_results.csv              -> Diebold-Mariano results
  artifacts/metrics/hac_lag_sensitivity.csv          -> HAC lag sensitivity
  artifacts/metrics/checkpoint_integrity_manifest.csv -> checkpoint completeness
  src/dataset.py (FEATURE_SETS), src/models/*        -> methodology facts

No ML pipeline code is modified. No metric is invented: if an artifact is
missing, the corresponding field is emitted as null and the UI shows it as
unavailable.

Usage:
    python app/build_dashboard_data.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METRICS = ROOT / "artifacts" / "metrics"
CKPTS = ROOT / "artifacts" / "checkpoints"
OUT = ROOT / "app" / "dashboard" / "data.js"

from src.dataset import FEATURE_SETS  # noqa: E402
from src.evaluate import compute_forecasting_metrics  # noqa: E402


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(v):
    """Convert numpy / NaN values into JSON-safe Python values."""
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def metric_block(m: dict | None) -> dict | None:
    if not m:
        return None
    keys = ["rmse_kw", "mae_kw", "wape_pct", "mape_pct", "r2_score",
            "nrmse_pct", "persistence_skill_score", "peak_mae_kw"]
    return {k: clean(m.get(k)) for k in keys}


def param_counts() -> dict | None:
    """Trainable parameter counts computed from the real model classes."""
    try:
        from src.models import CNN1DForecaster, LSTMForecaster, CNNLSTMForecaster
    except Exception:
        return None
    out = {}
    for name, cls in [("1D_CNN", CNN1DForecaster), ("LSTM", LSTMForecaster),
                      ("CNN_LSTM", CNNLSTMForecaster)]:
        out[name] = {}
        for n_feat in sorted({len(v) for v in FEATURE_SETS.values()}):
            try:
                m = cls(in_features=n_feat)
                out[name][str(n_feat)] = int(sum(p.numel() for p in m.parameters() if p.requires_grad))
            except Exception:
                out[name][str(n_feat)] = None
    return out


def main() -> None:
    sources = {}
    warnings: list[str] = []

    # ------------------------------------------------------------------ #
    # 1. Experiment records
    # ------------------------------------------------------------------ #
    tr_path = METRICS / "training_results.json"
    records = json.loads(tr_path.read_text(encoding="utf-8"))
    sources["training_results.json"] = sha256(tr_path)

    status_map = {}
    cons_path = METRICS / "phase_3_4_consolidated_results.csv"
    if cons_path.exists():
        cons = pd.read_csv(cons_path)
        status_map = dict(zip(cons["experiment_id"], cons["statistical_validation_status"]))
        sources["phase_3_4_consolidated_results.csv"] = sha256(cons_path)

    manifest = {}
    man_path = METRICS / "checkpoint_integrity_manifest.csv"
    if man_path.exists():
        man = pd.read_csv(man_path)
        manifest = dict(zip(man["experiment_id"], man["type"]))
        sources["checkpoint_integrity_manifest.csv"] = sha256(man_path)

    agg_pred_path = METRICS / "aggregate_test_predictions.parquet"
    preds = pd.read_parquet(agg_pred_path) if agg_pred_path.exists() else None
    if preds is not None:
        sources["aggregate_test_predictions.parquet"] = sha256(agg_pred_path)
    pred_exp_ids = set(preds["exp_id"].unique()) if preds is not None else set()

    experiments = []
    per_meter = {}
    for r in records:
        exp_id = r["exp_id"]
        scale = "aggregate" if r["cohort_type"] == "aggregate" else "individual"
        ckpt_file = CKPTS / f"{exp_id}.pt"
        ckpt_state = "missing"
        if ckpt_file.exists():
            ckpt_state = "partial" if "partial" in str(manifest.get(exp_id, "")) else "available"

        md = r.get("meter_distribution") or {}
        if md.get("per_meter_metrics"):
            per_meter[exp_id] = {
                mid: {k: clean(v.get(k)) for k in
                      ["rmse_kw", "mae_kw", "wape_pct", "r2_score", "persistence_skill_score"]}
                for mid, v in md["per_meter_metrics"].items()
            }

        experiments.append({
            "id": exp_id,
            "scale": scale,
            "district": r["district"],
            "arch": r["model"],
            "features": r["feature_config"],
            "in_features": r.get("in_features"),
            "best_epoch": r.get("best_epoch"),
            "total_epochs": r.get("total_epochs"),
            "train_sec": clean(r.get("training_duration_sec")),
            "best_val_loss": clean(r.get("best_val_loss")),
            "metrics": metric_block(r.get("metrics")),
            "meter_summary": {k: clean(v) for k, v in md.items() if k != "per_meter_metrics"} or None,
            "validation_status": status_map.get(exp_id),
            "checkpoint": ckpt_state,
            "pointwise_predictions": exp_id in pred_exp_ids,
        })

    # ------------------------------------------------------------------ #
    # 2. Persistence baselines + census
    # ------------------------------------------------------------------ #
    bb_path = METRICS / "baseline_benchmarks.json"
    bb = json.loads(bb_path.read_text(encoding="utf-8"))
    sources["baseline_benchmarks.json"] = sha256(bb_path)
    baselines, census = {}, {}
    for district, block in bb.get("aggregates", {}).items():
        fe = block.get("full_test_evaluation", {})
        baselines[f"aggregate|{district}"] = metric_block(fe.get("naive_persistence"))
        census[f"aggregate|{district}"] = {k: clean(v) for k, v in block.get("census", {}).items()}
    ind_key = next((k for k in bb if k not in ("metadata", "aggregates")), None)
    for district, block in (bb.get(ind_key, {}) if ind_key else {}).items():
        baselines[f"individual|{district}"] = metric_block(block.get("pooled"))
        c = {k: clean(v) for k, v in block.get("census", {}).items()}
        c["meter_count"] = block.get("meter_count")
        census[f"individual|{district}"] = c

    # ------------------------------------------------------------------ #
    # 3. Pointwise aggregate predictions (compact, per district)
    #    + reconciliation against logged metrics
    # ------------------------------------------------------------------ #
    series = {}
    reconciliation = []
    if preds is not None:
        for district, g in preds.groupby("district"):
            g = g.sort_values(["exp_id", "timestamp"])
            first = g[g["exp_id"] == g["exp_id"].iloc[0]]
            ts = first["timestamp"].dt.strftime("%Y-%m-%dT%H:%M").tolist()
            series[district] = {
                "timestamps": ts,
                "actual": [round(float(x), 4) for x in first["actual_demand_kw"]],
                "persistence": [round(float(x), 4) for x in first["naive_persistence_kw"]],
                "predicted": {},
            }
            for exp_id, ge in g.groupby("exp_id"):
                if ge["timestamp"].dt.strftime("%Y-%m-%dT%H:%M").tolist() != ts:
                    warnings.append(f"{exp_id}: timestamps not aligned with district series; skipped")
                    continue
                series[district]["predicted"][exp_id] = [round(float(x), 4) for x in ge["predicted_demand_kw"]]
                m = compute_forecasting_metrics(ge["actual_demand_kw"].values,
                                                ge["predicted_demand_kw"].values,
                                                ge["naive_persistence_kw"].values)
                logged = next((e["metrics"] for e in experiments if e["id"] == exp_id), None) or {}
                d_rmse = abs(m["rmse_kw"] - logged.get("rmse_kw", np.nan)) if logged.get("rmse_kw") is not None else None
                d_mae = abs(m["mae_kw"] - logged.get("mae_kw", np.nan)) if logged.get("mae_kw") is not None else None
                reconciliation.append({
                    "id": exp_id, "n": int(len(ge)),
                    "rmse_recomputed": m["rmse_kw"], "rmse_logged": logged.get("rmse_kw"),
                    "mae_recomputed": m["mae_kw"], "mae_logged": logged.get("mae_kw"),
                    "abs_diff_rmse": clean(d_rmse), "abs_diff_mae": clean(d_mae),
                    "match": bool(d_rmse is not None and d_rmse < 1e-3 and d_mae is not None and d_mae < 1e-3),
                })

    # ------------------------------------------------------------------ #
    # 4. Statistical tests
    # ------------------------------------------------------------------ #
    dm = []
    dm_path = METRICS / "dm_test_results.csv"
    if dm_path.exists():
        sources["dm_test_results.csv"] = sha256(dm_path)
        for _, row in pd.read_csv(dm_path).iterrows():
            dm.append({
                "district": row["district"], "scale": row["cohort_type"],
                "comparison": row["comparison_name"], "category": row["category"],
                "model_1": row["model_1"], "model_2": row["model_2"],
                "dm_stat": clean(row["dm_stat"]), "p_raw": clean(row["p_value_raw"]),
                "p_adj": clean(row["p_value_adj"]), "significant": bool(row["significant_fwer_05"]),
                "effect": row["effect_direction"], "n": clean(row["sample_size"]),
                "mse_1": clean(row["loss_model1"]), "mse_2": clean(row["loss_model2"]),
            })
    hac = []
    hac_path = METRICS / "hac_lag_sensitivity.csv"
    if hac_path.exists():
        sources["hac_lag_sensitivity.csv"] = sha256(hac_path)
        h = pd.read_csv(hac_path)
        lags = sorted({int(c.split("_K")[1]) for c in h.columns if c.startswith("DM_K")})
        for _, row in h.iterrows():
            hac.append({"district": row["district"], "hypothesis": row["hypothesis"],
                        "by_lag": {str(k): {"dm": clean(row[f"DM_K{k}"]),
                                            "sig": bool(row[f"sig_K{k}"])} for k in lags}})

    # ------------------------------------------------------------------ #
    # 5. Methodology facts (read from source code where possible)
    # ------------------------------------------------------------------ #
    meta = bb.get("metadata", {})
    methodology = {
        "lookback_steps": meta.get("lookback_len"),
        "lookback_hours": meta.get("lookback_hours"),
        "horizon_steps": meta.get("horizon_steps"),
        "horizon_minutes": meta.get("horizon_minutes"),
        "split": meta.get("train_val_test_split"),
        "feature_sets": FEATURE_SETS,
        "param_counts": param_counts(),
        "training": {  # values as written in src/train_runner.py
            "loss": "HuberLoss(delta=1.0) on standardised target",
            "optimizer": "AdamW, lr=1e-3, weight_decay=1e-4",
            "grad_clip": "clip_grad_norm_(max_norm=1.0)",
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6)",
            "early_stopping": "patience 8 (aggregate), 6 (individual); best-validation weights restored",
            "epochs": "max 40 (aggregate), 15 (individual)",
            "batch_size": "64 (aggregate), 512 (individual)",
            "source": "src/train_runner.py",
        },
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources,
        "warnings": warnings,
        "experiments": experiments,
        "baselines": baselines,
        "census": census,
        "series": series,
        "per_meter": per_meter,
        "reconciliation": reconciliation,
        "dm_tests": dm,
        "hac_sensitivity": hac,
        "methodology": methodology,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("// AUTO-GENERATED by app/build_dashboard_data.py — do not edit by hand.\n"
                   "window.DASHBOARD_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
                   encoding="utf-8")

    n_match = sum(r["match"] for r in reconciliation)
    print(f"Experiments: {len(experiments)} | aggregate series: {list(series)} | "
          f"reconciled {n_match}/{len(reconciliation)} | per-meter blocks: {len(per_meter)}")
    for w in warnings:
        print("WARNING:", w)
    print(f"Wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()

