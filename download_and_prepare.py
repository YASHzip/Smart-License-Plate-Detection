"""
download_and_prepare.py - One-Shot Dataset Download & Preparation

Downloads all required Indian plate datasets via kagglehub (no browser needed,
no API key needed for public datasets) then runs the full preparation pipeline.

Datasets downloaded:
  1. abtexp/synthetic-indian-license-plates  - 18,000 crops, ALL 36 states
  2. kedarsai/indian-license-plates-with-labels - ~2,000 real Indian plates (YOLO)

Then automatically:
  - Generates 4,000 synthetic training images (all-state coverage)
  - Merges real + synthetic + existing archive/ into indian_dataset/
  - Writes indian_data.yaml ready for fine-tuning

Usage:
    pip install kagglehub          # one-time install
    python download_and_prepare.py

    # Skip synthetic generation (faster, real plates only):
    python download_and_prepare.py --no-synthetic

    # Custom synthetic image count:
    python download_and_prepare.py --synth-count 6000
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()


# ── kagglehub download ────────────────────────────────────────────────────────

def download_dataset(handle: str, cache_dir: str = None) -> Path:
    """Download a Kaggle dataset via kagglehub and return its local path."""
    import os
    # Redirect cache to D: drive if C: is full or cache_dir specified
    if cache_dir:
        os.environ["KAGGLE_CACHE_DIR"] = cache_dir
        os.environ["KAGGLEHUB_CACHE"] = cache_dir

    try:
        import kagglehub
    except ImportError:
        print("[ERROR] kagglehub not installed.")
        print("        Run:  pip install kagglehub")
        sys.exit(1)

    print(f"\n[DOWNLOAD] {handle} ...")
    path = kagglehub.dataset_download(handle)
    print(f"[DOWNLOAD] Saved to: {path}")
    return Path(path)


# ── subprocess helpers ────────────────────────────────────────────────────────

def run(cmd: list, cwd=None):
    """Run a command and stream its output. Exit on failure."""
    print(f"\n[RUN] {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, cwd=cwd or BASE_DIR)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Download datasets and prepare Indian plate training data")
    ap.add_argument("--no-synthetic", action="store_true",
                    help="Skip synthetic image generation (use real data only)")
    ap.add_argument("--synth-count", type=int, default=4000,
                    help="Number of synthetic images to generate (default: 4000)")
    ap.add_argument("--synth-per-image", type=int, default=1,
                    help="Plates per synthetic image 1-3 (default: 1)")
    ap.add_argument("--cache-dir", type=str,
                    default=r"D:\kaggle_cache",
                    help=r"Kaggle download cache dir (default: D:\kaggle_cache)")
    ap.add_argument("--epochs", type=int, default=80,
                    help="Training epochs — printed in final instructions (default: 80)")
    args = ap.parse_args()

    print("\n" + "=" * 62)
    print("  Indian License Plate - Full Dataset Download & Prep")
    print("=" * 62)

    # ── Step 1: Download datasets ─────────────────────────────────────────────
    print("\n[STEP 1/4] Downloading datasets via kagglehub ...")
    print(f"           Cache dir -> {args.cache_dir}  (D: has 241 GB free)")

    synthetic_path = download_dataset("abtexp/synthetic-indian-license-plates",
                                      cache_dir=args.cache_dir)
    real_path      = download_dataset("kedarsai/indian-license-plates-with-labels",
                                      cache_dir=args.cache_dir)

    # ── Step 2: Generate synthetic training images ────────────────────────────
    if not args.no_synthetic:
        print(f"\n[STEP 2/4] Generating {args.synth_count} synthetic "
              f"training images (all 36 states) ...")
        run([
            sys.executable,
            str(BASE_DIR / "generate_synthetic_data.py"),
            "--plates",           str(synthetic_path),
            "--output",           str(BASE_DIR / "synthetic_dataset"),
            "--count",            str(args.synth_count),
            "--plates-per-image", str(args.synth_per_image),
        ])
    else:
        print("\n[STEP 2/4] Skipped synthetic generation (--no-synthetic).")

    # ── Step 3: Merge datasets ────────────────────────────────────────────────
    print("\n[STEP 3/4] Merging real Indian plate dataset ...")
    run([
        sys.executable,
        str(BASE_DIR / "prepare_indian_dataset.py"),
        "--new-dataset", str(real_path),
    ])

    if not args.no_synthetic:
        print("\n[STEP 3b/4] Merging synthetic dataset ...")
        run([
            sys.executable,
            str(BASE_DIR / "prepare_indian_dataset.py"),
            "--new-dataset", str(BASE_DIR / "synthetic_dataset"),
        ])

    # ── Step 4: Print training command ───────────────────────────────────────
    print("\n" + "=" * 62)
    print("  [STEP 4/4] Dataset ready!  Run this to fine-tune:")
    print("=" * 62)
    print(f"""
  cd yolov5
  python train.py ^
    --img 640 ^
    --batch 16 ^
    --epochs {args.epochs} ^
    --data ..\\indian_data.yaml ^
    --weights runs\\train\\exp2\\weights\\best.pt ^
    --hyp ..\\hyp_indian.yaml ^
    --project runs\\train ^
    --name indian_finetune ^
    --device 0 ^
    --cache
""")
    print("  After training, detection auto-uses the new model:")
    print("  python detect_and_recognize.py --source your_image.jpg")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
