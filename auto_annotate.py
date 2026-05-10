"""
auto_annotate.py - Auto-Annotation Tool

Uses the current best YOLOv5 model to detect license plates in a folder of
UNLABELLED images and saves YOLO-format .txt label files alongside them.

Useful for:
  - Labelling the ~2000 unlabelled images in kedarsai/indian-license-plates dataset
  - Quickly bootstrapping labels for any new raw image collection
  - Expanding the training set without manual labelling effort

Labels with confidence below --min-conf are discarded to keep quality high.
A summary CSV is written to the output folder for review.

Usage:
    python auto_annotate.py --images D:\\unlabelled_plates --output D:\\labelled_out
    python auto_annotate.py --images D:\\unlabelled_plates --min-conf 0.55
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch

BASE_DIR        = Path(__file__).parent.resolve()
YOLOV5_DIR      = BASE_DIR / "yolov5"
_INDIAN_WEIGHTS = YOLOV5_DIR / "runs" / "train" / "indian_finetune" / "weights" / "best.pt"
_ORIG_WEIGHTS   = YOLOV5_DIR / "runs" / "train" / "exp2"            / "weights" / "best.pt"
DEFAULT_WEIGHTS = str(_INDIAN_WEIGHTS if _INDIAN_WEIGHTS.exists() else _ORIG_WEIGHTS)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(weights: str, conf: float, iou: float):
    print(f"[INFO] Loading model: {weights}")
    model = torch.hub.load(
        str(YOLOV5_DIR), "custom",
        path=weights, source="local", verbose=False
    )
    model.conf = conf
    model.iou  = iou
    model.eval()
    device = "GPU" if next(model.parameters()).is_cuda else "CPU"
    print(f"[INFO] Model ready on {device}")
    return model


def annotate_image(model, img_path: Path, min_conf: float) -> list:
    """
    Run detection on one image.
    Returns list of YOLO-format strings '0 cx cy bw bh'.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    H, W = img.shape[:2]
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results    = model(rgb)
    detections = results.xyxy[0].cpu().numpy()   # [x1,y1,x2,y2,conf,cls]

    lines = []
    for *box, conf, cls in detections:
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = [max(0, v) for v in map(int, box)]
        x2, y2 = min(x2, W), min(y2, H)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


def main():
    ap = argparse.ArgumentParser(
        description="Auto-annotate unlabelled images with YOLOv5 license plate detector")
    ap.add_argument("--images",   required=True,
                    help="Folder of unlabelled images to annotate")
    ap.add_argument("--output",   default=None,
                    help="Output folder for labels (default: same as images)")
    ap.add_argument("--weights",  default=DEFAULT_WEIGHTS,
                    help=f"Model weights path (default: {DEFAULT_WEIGHTS})")
    ap.add_argument("--min-conf", type=float, default=0.50,
                    help="Min detection confidence to accept (default: 0.50)")
    ap.add_argument("--iou",      type=float, default=0.45,
                    help="NMS IoU threshold (default: 0.45)")
    ap.add_argument("--img-size", type=int, default=640,
                    help="Inference image size (default: 640)")
    ap.add_argument("--copy-images", action="store_true",
                    help="Also copy images to --output folder")
    args = ap.parse_args()

    img_dir = Path(args.images)
    if not img_dir.exists():
        sys.exit(f"[ERROR] Images folder not found: {img_dir}")

    out_dir = Path(args.output) if args.output else img_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.weights).exists():
        sys.exit(f"[ERROR] Weights not found: {args.weights}")

    model    = load_model(args.weights, args.min_conf, args.iou)
    img_list = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

    if not img_list:
        sys.exit(f"[ERROR] No images found in {img_dir}")

    print(f"\n{'='*58}")
    print(f"  Auto-Annotation")
    print(f"{'='*58}")
    print(f"  Images found : {len(img_list)}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Min conf     : {args.min_conf}")
    print(f"{'='*58}\n")

    stats    = []
    labelled = 0
    skipped  = 0

    for i, img_path in enumerate(img_list, 1):
        lines = annotate_image(model, img_path, args.min_conf)

        lbl_path = out_dir / (img_path.stem + ".txt")

        if lines:
            lbl_path.write_text("\n".join(lines) + "\n")
            labelled += 1
        else:
            # Write empty label so YOLO knows the image has no plates
            lbl_path.write_text("")
            skipped += 1

        stats.append({
            "image":        img_path.name,
            "plates_found": len(lines),
            "label_file":   lbl_path.name,
        })

        if args.copy_images and args.output:
            import shutil
            shutil.copy2(img_path, out_dir / img_path.name)

        if i % 100 == 0 or i == len(img_list):
            print(f"  [{i:>5}/{len(img_list)}] "
                  f"labelled={labelled}  no-plate={skipped}")

    # Write summary CSV
    csv_path = out_dir / "annotation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "plates_found", "label_file"])
        writer.writeheader()
        writer.writerows(stats)

    print(f"\n{'='*58}")
    print(f"  Done!")
    print(f"  Images with plates : {labelled}")
    print(f"  Images without     : {skipped}")
    print(f"  Summary CSV        : {csv_path}")
    print(f"{'='*58}")
    print(f"\nNext steps:")
    print(f"  1. Review annotation_summary.csv")
    print(f"  2. Spot-check a few labels with a viewer")
    print(f"  3. Run prepare_indian_dataset.py --new-dataset {out_dir}")
    print(f"  4. Fine-tune the model\n")


if __name__ == "__main__":
    main()
