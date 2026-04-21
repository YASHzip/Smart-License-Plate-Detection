"""
testing.py — Smart License Plate Detection
Phase A: Quick inference test on a single image.

Uses the best trained weights (best.pt) and accepts CLI arguments
so you don't need to hardcode paths.

Usage:
    python testing.py                                        # uses default test image
    python testing.py --image path/to/your_image.png        # custom image
    python testing.py --image img.jpg --conf 0.3 --no-save  # custom conf, no output save
"""

import argparse
import sys
from pathlib import Path

import torch
import cv2

# ─── Defaults ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
YOLOV5_DIR = BASE_DIR / "yolov5"

DEFAULT_WEIGHTS = YOLOV5_DIR / "runs" / "train" / "exp2" / "weights" / "best.pt"
DEFAULT_IMAGE   = BASE_DIR / "archive" / "images" / "test" / "Cars252.png"
DEFAULT_SAVE_DIR = BASE_DIR / "output_test_results"


def parse_args():
    parser = argparse.ArgumentParser(description="Quick inference test for the license plate detector.")
    parser.add_argument("--image",   type=str, default=str(DEFAULT_IMAGE),
                        help="Path to the test image")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                        help="Path to model weights (default: best.pt)")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou",     type=float, default=0.45,
                        help="IoU NMS threshold (default: 0.45)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save the output image")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display the result window")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not Path(args.weights).exists():
        print(f"[ERROR] Weights not found: {args.weights}")
        print("  Make sure the model has been trained.")
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"[ERROR] Image not found: {args.image}")
        print("  Usage: python testing.py --image path/to/image.png")
        sys.exit(1)

    print("=" * 60)
    print("  Smart License Plate Detection — Quick Test")
    print("=" * 60)
    print(f"  Image   : {args.image}")
    print(f"  Weights : {args.weights}")
    print(f"  Conf    : {args.conf}   IoU: {args.iou}")
    print("=" * 60)

    # ── Load model ───────────────────────────────────────────────────────────
    print("[INFO] Loading model …")
    model = torch.hub.load(
        str(YOLOV5_DIR), "custom",
        path=args.weights, source="local", verbose=False
    )
    model.conf = args.conf
    model.iou  = args.iou
    model.eval()
    device = "GPU" if next(model.parameters()).is_cuda else "CPU"
    print(f"[INFO] Model loaded on {device}.")

    # ── Load and run inference ────────────────────────────────────────────────
    image     = cv2.imread(args.image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("[INFO] Running inference …")
    results = model(image_rgb)

    # ── Print detections ──────────────────────────────────────────────────────
    dets = results.xyxy[0].cpu().numpy()
    print(f"\n  Detected {len(dets)} license plate(s):\n")
    for i, (*box, conf, cls) in enumerate(dets, 1):
        x1, y1, x2, y2 = map(int, box)
        print(f"  [{i}] BBox: ({x1},{y1}) → ({x2},{y2})   Confidence: {conf:.2%}")

    if len(dets) == 0:
        print("  No plates detected. Try lowering --conf threshold.")

    # ── Save output ───────────────────────────────────────────────────────────
    if not args.no_save:
        results.save(save_dir=str(DEFAULT_SAVE_DIR))
        print(f"\n[INFO] Result saved to: {DEFAULT_SAVE_DIR}")

    # ── Show window ───────────────────────────────────────────────────────────
    if not args.no_show:
        results.show()

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
