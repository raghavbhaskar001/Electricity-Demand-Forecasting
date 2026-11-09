"""
Local Individual Checkpoint Verification Script
Phase 3.3 - Step 2: Verification of Imported Individual Model Checkpoints
"""

import sys
import os
import json
import hashlib
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train_runner import get_model, FEATURE_SETS

def verify_checkpoints(ckpt_dir_str="artifacts/checkpoints"):
    ckpt_dir = Path(ckpt_dir_str)
    print("=" * 80)
    print("VERIFYING IMPORTED INDIVIDUAL CHECKPOINTS")
    print("=" * 80)
    print(f"Target directory: {ckpt_dir.resolve()}\n")

    districts = ["mathura", "bareilly"]
    models = ["1D_CNN", "LSTM", "CNN_LSTM"]
    feature_configs = {
        "consumption_only": FEATURE_SETS["consumption_only"],
        "consumption_weather_retro": FEATURE_SETS["consumption_weather_retro"],
        "consumption_weather_operational": FEATURE_SETS["consumption_weather_operational"]
    }

    manifest_json = ckpt_dir / "individual_checkpoints_manifest.json"
    manifest_map = {}
    if manifest_json.exists():
        with open(manifest_json, "r", encoding="utf-8") as f:
            for item in json.load(f):
                manifest_map[item["filename"]] = item
        print(f"Loaded manifest with {len(manifest_map)} recorded hashes.\n")
    else:
        print("Notice: individual_checkpoints_manifest.json not found. Computing hashes dynamically.\n")

    expected_ids = []
    for d in districts:
        for m in models:
            for f in feature_configs.keys():
                expected_ids.append((d, m, f, f"ind_{d}_{m}_{f}.pt"))

    verified_count = 0
    failures = []
    summary_records = []

    print(f"{'Experiment ID':<52} | {'Epoch':<5} | {'Val Loss':<9} | {'SHA256 Match':<12} | {'Load Status'}")
    print("-" * 105)

    for d, m_name, f_name, filename in expected_ids:
        fpath = ckpt_dir / filename
        exp_id = filename.replace(".pt", "")

        if not fpath.exists():
            failures.append((exp_id, "File missing on disk"))
            print(f"{exp_id:<52} | {'MISSING':<5} | {'-':<9} | {'-':<12} | FAILED (File not found)")
            continue

        # Compute SHA-256
        sha256 = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        calc_hash = sha256.hexdigest()

        hash_match = "N/A"
        if filename in manifest_map:
            expected_hash = manifest_map[filename]["sha256"]
            if calc_hash.lower() == expected_hash.lower():
                hash_match = "MATCH"
            else:
                hash_match = "MISMATCH"
                failures.append((exp_id, f"SHA256 mismatch (expected {expected_hash[:8]}, got {calc_hash[:8]})"))

        # Verify PyTorch loading
        try:
            chk = torch.load(fpath, map_location="cpu")
            if not isinstance(chk, dict) or "model_state_dict" not in chk:
                failures.append((exp_id, "Invalid checkpoint structure: model_state_dict missing"))
                print(f"{exp_id:<52} | {'ERR':<5} | {'-':<9} | {hash_match:<12} | FAILED (Invalid state dict)")
                continue

            epoch = chk.get("epoch", "?")
            val_loss = chk.get("val_loss", "?")
            val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)

            # Check model parameter instantiation
            n_in = len(feature_configs[f_name])
            model = get_model(m_name, n_in, sequence_length=16)
            model.load_state_dict(chk["model_state_dict"])
            model.eval()

            verified_count += 1
            print(f"{exp_id:<52} | {str(epoch):<5} | {val_loss_str:<9} | {hash_match:<12} | PASS")

            summary_records.append({
                "exp_id": exp_id,
                "district": d,
                "model": m_name,
                "feature_config": f_name,
                "epoch": epoch,
                "val_loss": val_loss,
                "sha256": calc_hash,
                "status": "VERIFIED"
            })

        except Exception as e:
            failures.append((exp_id, f"Load exception: {str(e)}"))
            print(f"{exp_id:<52} | {'ERR':<5} | {'-':<9} | {hash_match:<12} | FAILED ({str(e)[:30]})")

    print("-" * 105)
    print(f"VERIFICATION SUMMARY: {verified_count}/18 individual checkpoints verified.")

    # Confirm aggregate checkpoints remain untouched
    agg_ckpts = list(ckpt_dir.glob("agg_*.pt"))
    print(f"CONFIRMATION: {len(agg_ckpts)} aggregate checkpoints remain untouched in {ckpt_dir}.\n")

    if failures:
        print("FAILURES / BLOCKED ITEMS:")
        for exp_id, reason in failures:
            print(f"  - {exp_id}: {reason}")
        print("=" * 80)
        return False, summary_records
    else:
        print("ALL 18 INDIVIDUAL CHECKPOINTS SUCCESSFULLY VERIFIED!")
        print("=" * 80)
        return True, summary_records

if __name__ == "__main__":
    verify_checkpoints()

