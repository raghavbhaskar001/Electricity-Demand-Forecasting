"""
Colab Checkpoint Packaging & SHA-256 Manifest Generator
Run this script or notebook cell in Google Colab to package the 18 individual checkpoints.
"""

import os
import hashlib
import json
import zipfile
from pathlib import Path

def package_individual_checkpoints(ckpt_dir_str="artifacts/checkpoints", output_zip="ind_checkpoints.zip", download=False):
    ckpt_dir = Path(ckpt_dir_str)
    
    # Expected 18 experiment IDs
    districts = ["mathura", "bareilly"]
    models = ["1D_CNN", "LSTM", "CNN_LSTM"]
    features = ["consumption_only", "consumption_weather_retro", "consumption_weather_operational"]
    
    expected_ids = []
    for d in districts:
        for m in models:
            for f in features:
                expected_ids.append(f"ind_{d}_{m}_{f}.pt")
                
    print("=" * 80)
    print("PACKAGING 18 INDIVIDUAL MODEL CHECKPOINTS")
    print("=" * 80)
    print(f"Target directory: {ckpt_dir.resolve()}")
    
    manifest = []
    files_to_pack = []
    
    for fname in expected_ids:
        fpath = ckpt_dir / fname
        if not fpath.exists():
            print(f"ERROR: Expected checkpoint {fname} missing from {ckpt_dir}!")
            continue
            
        # Compute SHA-256
        sha256 = hashlib.sha256()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        digest = sha256.hexdigest()
        size_bytes = fpath.stat().st_size
        
        manifest.append({
            "filename": fname,
            "exp_id": fname.replace(".pt", ""),
            "sha256": digest,
            "size_bytes": size_bytes
        })
        files_to_pack.append(fpath)
        print(f"  [OK] {fname:<55} ({size_bytes:>7} bytes) | SHA256: {digest[:12]}...")

    if len(manifest) != 18:
        print(f"\nWARNING: Found {len(manifest)}/18 expected individual checkpoints!")
    else:
        print(f"\nAll 18 individual checkpoints verified.")

    # Save manifest files
    manifest_json_path = ckpt_dir / "individual_checkpoints_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    manifest_csv_path = ckpt_dir / "individual_checkpoints_manifest.csv"
    with open(manifest_csv_path, "w", encoding="utf-8") as f:
        f.write("filename,exp_id,sha256,size_bytes\n")
        for item in manifest:
            f.write(f"{item['filename']},{item['exp_id']},{item['sha256']},{item['size_bytes']}\n")

    # Create ZIP
    print(f"\nCreating ZIP archive: {output_zip} ...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fpath in files_to_pack:
            zipf.write(fpath, arcname=fpath.name)
        zipf.write(manifest_json_path, arcname="individual_checkpoints_manifest.json")
        zipf.write(manifest_csv_path, arcname="individual_checkpoints_manifest.csv")
        
    zip_size = Path(output_zip).stat().st_size
    print(f"SUCCESS: Created {output_zip} ({zip_size / (1024 * 1024):.2f} MB)")
    print(f"Contains 18 individual checkpoints + SHA-256 manifests.")
    print("=" * 80)
    
    # Try Google Colab automatic download only if requested
    if download:
        try:
            from google.colab import files
            print("Triggering automatic browser download in Google Colab...")
            files.download(output_zip)
        except ImportError:
            print("Not running in Colab environment or Colab files module unavailable.")
            print(f"Please manually download or copy {output_zip} to artifacts/checkpoints/")
    else:
        print(f"Archive {output_zip} prepared successfully on disk (download deferred to consolidated bundle).")

if __name__ == "__main__":
    package_individual_checkpoints(download=True)

