"""
evaluate.py — Smart License Plate Detection
Phase A: Testing & Evaluation

Runs the trained YOLOv5 best.pt model on the test set,
computes mAP@0.5, Precision, Recall, F1 score,
and saves an evaluation report + annotated sample images.

Usage:
    python evaluate.py
    python evaluate.py --weights path/to/best.pt --data data.yaml --img 416
"""

import argparse
import os
import sys
import csv
import time
import subprocess
from pathlib import Path
from datetime import datetime

import torch
import cv2
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
YOLOV5_DIR   = BASE_DIR / "yolov5"
DEFAULT_WEIGHTS = YOLOV5_DIR / "runs" / "train" / "exp2" / "weights" / "best.pt"
DEFAULT_DATA    = BASE_DIR / "data.yaml"
REPORT_PATH     = BASE_DIR / "evaluation_report.txt"
EVAL_OUTPUT_DIR = BASE_DIR / "evaluation_output"


# ─── Argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the YOLOv5 license plate model on the test set.")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                        help="Path to model weights (default: best.pt)")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                        help="Path to data.yaml (default: project data.yaml)")
    parser.add_argument("--img", type=int, default=416,
                        help="Image size for inference (default: 416)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of sample images to save with annotations (default: 5)")
    parser.add_argument("--skip-val", action="store_true",
                        help="Skip val.py run and just display cached results.csv stats")
    return parser.parse_args()


# ─── Read best metrics from training results.csv ──────────────────────────────
def read_training_results():
    """Parse the results.csv from exp2 to get best epoch metrics."""
    results_csv = YOLOV5_DIR / "runs" / "train" / "exp2" / "results.csv"
    if not results_csv.exists():
        return None

    best = {"mAP_0.5": 0.0}
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys
            row = {k.strip(): v.strip() for k, v in row.items()}
            try:
                map50 = float(row.get("metrics/mAP_0.5", 0))
                if map50 > best["mAP_0.5"]:
                    best = {
                        "epoch":      int(float(row.get("epoch", 0))),
                        "precision":  float(row.get("metrics/precision", 0)),
                        "recall":     float(row.get("metrics/recall", 0)),
                        "mAP_0.5":   map50,
                        "mAP_0.5:0.95": float(row.get("metrics/mAP_0.5:0.95", 0)),
                        "box_loss":   float(row.get("val/box_loss", 0)),
                        "obj_loss":   float(row.get("val/obj_loss", 0)),
                    }
            except (ValueError, KeyError):
                continue
    return best


# ─── Run YOLOv5 val.py for test set evaluation ────────────────────────────────
def run_val(args):
    """Call yolov5/val.py as a subprocess to evaluate on the test split."""
    val_script = YOLOV5_DIR / "val.py"
    if not val_script.exists():
        print(f"[ERROR] val.py not found at {val_script}")
        return None

    print("\n[INFO] Running YOLOv5 val.py on test set …")
    cmd = [
        sys.executable, str(val_script),
        "--weights", args.weights,
        "--data",    args.data,
        "--img",     str(args.img),
        "--conf-thres", str(args.conf),
        "--iou-thres",  str(args.iou),
        "--task",    "test",
        "--save-txt",
        "--save-conf",
        "--project", str(BASE_DIR / "yolov5" / "runs" / "val"),
        "--name",    "eval",
        "--exist-ok",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(YOLOV5_DIR))
        print(result.stdout)
        if result.returncode != 0:
            print("[WARNING] val.py stderr:", result.stderr[-2000:])
        return result.stdout
    except Exception as e:
        print(f"[ERROR] Failed to run val.py: {e}")
        return None


# ─── Save annotated sample images ─────────────────────────────────────────────
def save_sample_detections(args, n_samples=5):
    """Run inference on a few test images and save annotated results."""
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    # Locate test images
    data_yaml_dir = Path(args.data).parent
    test_img_dir  = data_yaml_dir / "archive" / "images" / "test"

    # Fallback: look in drive E or D
    if not test_img_dir.exists():
        for drive in ["E:", "D:"]:
            candidate = Path(drive) / "License Plate Number Detection" / "archive" / "images" / "test"
            if candidate.exists():
                test_img_dir = candidate
                break

    if not test_img_dir.exists():
        print(f"[WARNING] Test image directory not found. Skipping sample saves.")
        return

    images = list(test_img_dir.glob("*.png")) + list(test_img_dir.glob("*.jpg"))
    if not images:
        print("[WARNING] No test images found.")
        return

    # Load model
    print(f"\n[INFO] Loading model from {args.weights} for sample detections …")
    try:
        model = torch.hub.load(str(YOLOV5_DIR), "custom",
                               path=args.weights, source="local")
        model.conf = args.conf
        model.iou  = args.iou
        model.eval()
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    saved = 0
    for img_path in images[:n_samples]:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = model(img_rgb, size=args.img)
        detections = results.xyxy[0].cpu().numpy()

        # Draw boxes
        annotated = img_bgr.copy()
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"Plate {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out_path = EVAL_OUTPUT_DIR / f"sample_{saved+1}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"  [+] Saved annotated sample: {out_path.name}")
        saved += 1

    print(f"\n[INFO] {saved} sample image(s) saved to: {EVAL_OUTPUT_DIR}")


# ─── Write evaluation report ──────────────────────────────────────────────────
def write_report(best_train_metrics, val_output):
    """Save a formatted evaluation report to evaluation_report.txt."""
    sep   = "=" * 60
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        sep,
        "  SMART LICENSE PLATE DETECTION — EVALUATION REPORT",
        sep,
        f"  Generated : {now}",
        f"  Model     : YOLOv5s (custom fine-tuned)",
        f"  Weights   : runs/train/exp2/weights/best.pt",
        f"  Dataset   : Kaggle Car Plate Detection",
        f"  Classes   : 1 (license_plate)",
        sep,
        "",
        "  ── TRAINING SUMMARY (best epoch from results.csv) ──",
    ]

    if best_train_metrics:
        p  = best_train_metrics.get("precision", 0)
        r  = best_train_metrics.get("recall", 0)
        f1 = 2 * p * r / (p + r + 1e-9)
        lines += [
            f"  Epoch          : {best_train_metrics.get('epoch', 'N/A')} / 99",
            f"  Precision      : {p:.4f}  ({p*100:.1f}%)",
            f"  Recall         : {r:.4f}  ({r*100:.1f}%)",
            f"  F1 Score       : {f1:.4f}  ({f1*100:.1f}%)",
            f"  mAP@0.5        : {best_train_metrics.get('mAP_0.5', 0):.4f}  "
            f"({best_train_metrics.get('mAP_0.5', 0)*100:.1f}%)",
            f"  mAP@0.5:0.95   : {best_train_metrics.get('mAP_0.5:0.95', 0):.4f}  "
            f"({best_train_metrics.get('mAP_0.5:0.95', 0)*100:.1f}%)",
            f"  Val Box Loss   : {best_train_metrics.get('box_loss', 0):.6f}",
            f"  Val Obj Loss   : {best_train_metrics.get('obj_loss', 0):.6f}",
        ]
    else:
        lines.append("  [results.csv not found — training results unavailable]")

    lines += [
        "",
        "  ── TEST SET EVALUATION (val.py output) ──",
    ]
    if val_output:
        # Extract key lines from val.py stdout
        for line in val_output.splitlines():
            stripped = line.strip()
            if any(kw in stripped.lower() for kw in ["map", "precision", "recall", "all", "speed"]):
                lines.append(f"  {stripped}")
    else:
        lines.append("  [val.py was skipped or failed — run without --skip-val for full test metrics]")

    lines += [
        "",
        sep,
        "  Sample annotated images saved to: evaluation_output/",
        sep,
    ]

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n[INFO] Report saved to: {REPORT_PATH}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("=" * 60)
    print("  Smart License Plate Detection — Evaluation")
    print("=" * 60)
    print(f"  Weights : {args.weights}")
    print(f"  Data    : {args.data}")
    print(f"  Img size: {args.img}")
    print(f"  Conf    : {args.conf}   IoU: {args.iou}")
    print("=" * 60)

    # Step 1: Read training results
    best_train = read_training_results()

    # Step 2: Run val.py on test split (unless skipped)
    val_out = None
    if not args.skip_val:
        val_out = run_val(args)
    else:
        print("\n[INFO] --skip-val flag set. Skipping val.py run.")

    # Step 3: Save sample annotated images
    save_sample_detections(args, n_samples=args.samples)

    # Step 4: Write report
    write_report(best_train, val_out)


if __name__ == "__main__":
    main()
