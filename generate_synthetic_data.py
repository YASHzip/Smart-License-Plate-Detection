"""
generate_synthetic_data.py - Synthetic Training Data Generator

Takes Indian plate crop images (e.g. from abtexp/synthetic-indian-license-plates
on Kaggle — 18,000 images covering ALL 36 Indian states) and pastes them onto
background road/car images with random perspective, lighting and blur.
Auto-generates YOLO bounding-box labels.

This solves the "only one state" problem: synthetic plates from ALL states are
used so the detection model sees every Indian plate colour/style combination.

Usage:
    python generate_synthetic_data.py ^
        --plates   D:\\synthetic_plates ^
        --output   synthetic_dataset ^
        --count    4000

    # Use custom backgrounds instead of archive/:
    python generate_synthetic_data.py ^
        --plates      D:\\synthetic_plates ^
        --backgrounds D:\\road_images ^
        --output      synthetic_dataset ^
        --count       4000
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR          = Path(__file__).parent.resolve()
DEFAULT_BG_DIR    = BASE_DIR / "archive" / "images" / "train"
IMG_EXTS          = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Image helpers ─────────────────────────────────────────────────────────────

def _imgs(directory: Path) -> list:
    return [p for p in directory.rglob("*") if p.suffix.lower() in IMG_EXTS]


def _perspective(img: np.ndarray, max_skew: float = 0.12) -> np.ndarray:
    """Random 4-point perspective warp to simulate viewing angle."""
    h, w = img.shape[:2]
    dx, dy = int(w * max_skew), int(h * max_skew)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, dx),     random.randint(0, dy)],
        [w - random.randint(0, dx), random.randint(0, dy)],
        [w - random.randint(0, dx), h - random.randint(0, dy)],
        [random.randint(0, dx),     h - random.randint(0, dy)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _photometric(img: np.ndarray) -> np.ndarray:
    """Random brightness/contrast + optional blur + optional noise."""
    alpha = random.uniform(0.55, 1.45)
    beta  = random.randint(-55, 55)
    img   = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.35:
        k = random.choice([3, 5])
        h, w = img.shape[:2]
        if h >= k and w >= k:
            img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < 0.20:
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def _load_existing_labels(bg_path: Path) -> list:
    """Try to read YOLO .txt labels that correspond to a background image."""
    lbl = bg_path.parent.parent.parent / "labels" / \
          bg_path.parent.name / (bg_path.stem + ".txt")
    rows = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                rows.append([float(p) for p in parts])
    return rows


# ── Core paste logic ──────────────────────────────────────────────────────────

def paste_plate(bg: np.ndarray, plate: np.ndarray,
                w_frac_range=(0.07, 0.22)) -> tuple:
    """
    Resize, distort, adjust and paste a plate crop onto the background.

    Returns:
        (result_bgr, (x1, y1, x2, y2)) or (None, None) on failure.
    """
    H, W = bg.shape[:2]

    # --- Resize plate to a realistic fraction of the frame width ---
    target_w = max(int(W * random.uniform(*w_frac_range)), 40)
    # Standard Indian plate aspect ratio ≈ 4.7 : 1 (522mm �-- 110mm)
    target_h = max(int(target_w / 4.7), 12)
    plate_r  = cv2.resize(plate, (target_w, target_h),
                          interpolation=cv2.INTER_LANCZOS4)

    # --- Augment (guard against corrupt/degenerate images) ---
    try:
        plate_r = _perspective(plate_r, max_skew=0.10)
        plate_r = _photometric(plate_r)
    except Exception:
        return None, None

    ph, pw = plate_r.shape[:2]
    if pw >= W or ph >= H or pw < 5 or ph < 3:
        return None, None

    # --- Position: lower 55 % of frame, away from edges ---
    margin = max(6, int(W * 0.01))
    x = random.randint(margin, W - pw - margin)
    y = random.randint(int(H * 0.40), H - ph - margin)
    x, y = max(0, min(x, W - pw)), max(0, min(y, H - ph))

    result              = bg.copy()
    result[y:y+ph, x:x+pw] = plate_r
    return result, (x, y, x + pw, y + ph)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic Indian plate training images")
    ap.add_argument("--plates",       required=True,
                    help="Folder with Indian plate crop images (all states)")
    ap.add_argument("--backgrounds",  default=str(DEFAULT_BG_DIR),
                    help=f"Folder with background images (default: {DEFAULT_BG_DIR})")
    ap.add_argument("--output",       default="synthetic_dataset",
                    help="Output root folder (default: synthetic_dataset)")
    ap.add_argument("--count",        type=int, default=4000,
                    help="Number of synthetic images to generate (default: 4000)")
    ap.add_argument("--plates-per-image", type=int, default=1,
                    help="Max plates pasted per image 1-3 (default: 1)")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    plate_paths = _imgs(Path(args.plates))
    bg_paths    = _imgs(Path(args.backgrounds))

    if not plate_paths:
        sys.exit(f"[ERROR] No images found in plates dir: {args.plates}")
    if not bg_paths:
        sys.exit(f"[ERROR] No images found in backgrounds dir: {args.backgrounds}")

    out_root  = Path(args.output)
    img_out   = out_root / "images" / "train"
    lbl_out   = out_root / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Synthetic Data Generator")
    print(f"{'='*60}")
    print(f"  Plate crops  : {len(plate_paths):>6}")
    print(f"  Backgrounds  : {len(bg_paths):>6}")
    print(f"  Target count : {args.count:>6}")
    print(f"  Output       : {out_root}")
    print(f"{'='*60}\n")

    try:
        from tqdm import tqdm
        pbar = tqdm(total=args.count, unit="img")
    except ImportError:
        pbar = None

    generated, failed = 0, 0

    while generated < args.count:
        bg_path = random.choice(bg_paths)
        bg      = cv2.imread(str(bg_path))
        if bg is None:
            failed += 1
            continue

        existing_boxes = _load_existing_labels(bg_path)
        H, W           = bg.shape[:2]
        result         = bg.copy()
        all_boxes      = list(existing_boxes)
        added          = 0

        n_plates = random.randint(1, max(1, args.plates_per_image))
        for _ in range(n_plates):
            plate = cv2.imread(str(random.choice(plate_paths)))
            if plate is None:
                continue
            modified, bbox = paste_plate(result, plate)
            if modified is None:
                continue
            result = modified
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            all_boxes.append([0, cx, cy, bw, bh])
            added += 1

        if added == 0:
            failed += 1
            if failed > args.count * 0.15:
                print("\n[ERROR] Too many failures — check input paths.")
                break
            continue

        stem    = f"synth_{generated:06d}"
        out_img = img_out / f"{stem}.jpg"
        out_lbl = lbl_out / f"{stem}.txt"

        cv2.imwrite(str(out_img), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(out_lbl, "w") as f:
            for b in all_boxes:
                f.write(f"0 {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

        generated += 1
        if pbar:
            pbar.update(1)
        elif generated % 500 == 0:
            print(f"  Generated {generated}/{args.count} ...")

    if pbar:
        pbar.close()

    print(f"\n[DONE] Generated {generated} images → {out_root}")
    print(f"       Failed/skipped: {failed}")
    print(f"\nNext: add to dataset by re-running prepare_indian_dataset.py")
    print(f"      with --new-dataset {out_root}\n")


if __name__ == "__main__":
    main()
