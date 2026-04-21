"""
detect_and_recognize.py — Smart License Plate Detection
Phase D: Full End-to-End Pipeline

Detects license plates using YOLOv5 and extracts plate numbers via EasyOCR.
Supports image, video, and real-time webcam input.
Detected plates are automatically saved to the SQLite database.

Usage:
    python detect_and_recognize.py --source Cars252.png          # single image
    python detect_and_recognize.py --source road_video.mp4       # video file
    python detect_and_recognize.py --source 0                    # webcam (index 0)
    python detect_and_recognize.py --source 0 --no-save          # webcam, don't save output
    python detect_and_recognize.py --source image.jpg --no-db    # skip database saving
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import torch
import numpy as np

# ─── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
YOLOV5_DIR = BASE_DIR / "yolov5"
DEFAULT_WEIGHTS = YOLOV5_DIR / "runs" / "train" / "exp2" / "weights" / "best.pt"
OUTPUT_DIR = BASE_DIR / "detection_output"


# ─── Argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart License Plate Detection + Recognition Pipeline"
    )
    parser.add_argument("--source",  type=str, default="0",
                        help="Input source: image path, video path, or webcam index (default: 0)")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS),
                        help="Path to YOLOv5 model weights")
    parser.add_argument("--img",     type=int, default=416,
                        help="Inference image size (default: 416)")
    parser.add_argument("--conf",    type=float, default=0.40,
                        help="Detection confidence threshold (default: 0.40)")
    parser.add_argument("--iou",     type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save output images/video")
    parser.add_argument("--no-db",   action="store_true",
                        help="Do not save detections to database")
    parser.add_argument("--no-ocr",  action="store_true",
                        help="Disable OCR (detection bounding boxes only)")
    parser.add_argument("--show",    action="store_true", default=True,
                        help="Display output window (default: True for webcam/video)")
    parser.add_argument("--db",      type=str, default=str(BASE_DIR / "plates.db"),
                        help="Path to SQLite database file")
    return parser.parse_args()


# ─── Model loading ────────────────────────────────────────────────────────────
def load_model(weights_path: str, conf: float, iou: float):
    """Load the custom YOLOv5 model."""
    if not Path(weights_path).exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        print("        Please ensure the model is trained (yolov5/runs/train/exp2/weights/best.pt)")
        sys.exit(1)

    print(f"[INFO] Loading YOLOv5 model: {weights_path}")
    model = torch.hub.load(
        str(YOLOV5_DIR), "custom",
        path=weights_path, source="local", verbose=False
    )
    model.conf = conf
    model.iou  = iou
    model.eval()
    print(f"[INFO] Model loaded. Running on: {'GPU' if next(model.parameters()).is_cuda else 'CPU'}")
    return model


# ─── Inference on a single frame ──────────────────────────────────────────────
def process_frame(frame_bgr: np.ndarray,
                  model,
                  ocr_func,
                  db_save_func,
                  source_name: str,
                  args) -> tuple:
    """
    Run detection + OCR on one frame/image.

    Returns:
        annotated_frame (np.ndarray), list of detected plate dicts
    """
    frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results    = model(frame_rgb, size=args.img)
    detections = results.xyxy[0].cpu().numpy()   # [x1,y1,x2,y2,conf,cls]

    plates_found = []
    annotated    = frame_bgr.copy()

    for *box, det_conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        # Clamp to frame dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame_bgr[y1:y2, x1:x2]
        plate_text   = ""
        ocr_conf     = 0.0

        # ── OCR ───────────────────────────────────────────────────────────
        if not args.no_ocr and ocr_func is not None:
            ocr_result   = ocr_func(crop)
            plate_text   = ocr_result.get("text", "")
            ocr_conf     = ocr_result.get("confidence", 0.0)

        plates_found.append({
            "plate":       plate_text,
            "det_conf":    float(det_conf),
            "ocr_conf":    ocr_conf,
            "bbox":        (x1, y1, x2, y2),
            "source":      source_name,
        })

        # ── Save to database ───────────────────────────────────────────────
        if not args.no_db and db_save_func is not None and plate_text:
            db_save_func(
                plate_number=plate_text,
                raw_ocr_text=plate_text,
                image_path=source_name,
                detection_confidence=float(det_conf),
                ocr_confidence=ocr_conf,
                source=_source_type(source_name),
                db_path=args.db,
            )

        # ── Draw bounding box + label ──────────────────────────────────────
        color  = (0, 220, 0)
        lw     = max(2, int((frame_bgr.shape[0] + frame_bgr.shape[1]) / 600))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, lw)

        label_det  = f"{det_conf:.0%}"
        label_text = plate_text if plate_text else "No text"
        label_ocr  = f"{ocr_conf:.0%}" if plate_text else ""

        # Background for label
        label_full = f" {label_text}  det:{label_det}"
        if label_ocr:
            label_full += f"  ocr:{label_ocr}"

        (lw2, lh2), baseline = cv2.getTextSize(
            label_full, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_label = max(y1 - 6, lh2 + 4)
        cv2.rectangle(annotated,
                      (x1, y_label - lh2 - 4),
                      (x1 + lw2 + 4, y_label + baseline),
                      color, -1)
        cv2.putText(annotated, label_full,
                    (x1 + 2, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated, plates_found


def _source_type(source: str) -> str:
    """Determine source type label from source string."""
    try:
        int(source)
        return "webcam"
    except ValueError:
        ext = Path(source).suffix.lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            return "video"
        return "image"


# ─── OSD overlay (On-Screen Display) ─────────────────────────────────────────
def draw_osd(frame: np.ndarray, fps: float, plate_count: int, total_saved: int):
    """Draw FPS and plate count overlay on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Plates detected: {plate_count}", (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Total saved to DB: {total_saved}", (8, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1, cv2.LINE_AA)
    return frame


# ─── Source routing ───────────────────────────────────────────────────────────
def run_image(source, model, ocr_func, db_save_func, args):
    """Process a single image file."""
    img = cv2.imread(source)
    if img is None:
        print(f"[ERROR] Cannot read image: {source}")
        return

    print(f"[INFO] Processing image: {source}")
    annotated, plates = process_frame(img, model, ocr_func, db_save_func, source, args)

    _print_plates(plates)

    if not args.no_save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stem = Path(source).stem
        out  = OUTPUT_DIR / f"{stem}_detected.jpg"
        cv2.imwrite(str(out), annotated)
        print(f"[INFO] Saved: {out}")

    cv2.imshow("Smart License Plate Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video_or_webcam(source, model, ocr_func, db_save_func, args):
    """Process a video file or a live webcam feed."""
    # Try numeric webcam index
    try:
        cap_source = int(source)
        source_type = "webcam"
    except ValueError:
        cap_source  = source
        source_type = "video"

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        print("        For webcam, try --source 0 or --source 1")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w_in   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    writer = None
    out_path = None
    if not args.no_save and source_type == "video":
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stem     = Path(str(source)).stem
        out_path = OUTPUT_DIR / f"{stem}_detected.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps_in, (w_in, h_in))

    print(f"\n[INFO] Starting {source_type} stream …  Press Q to quit.\n")
    frame_idx   = 0
    total_saved = 0
    fps_display = 0.0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if source_type == "webcam":
                    print("[WARNING] Empty frame from webcam, retrying …")
                    time.sleep(0.05)
                    continue
                break   # end of video

            frame_idx += 1

            # Run detection every frame (or every N frames for speed)
            annotated, plates = process_frame(
                frame, model, ocr_func, db_save_func,
                f"{source_type}_frame_{frame_idx}", args
            )
            total_saved += len([p for p in plates if p["plate"]])

            # FPS calculation
            elapsed    = time.time() - t0
            fps_display = frame_idx / elapsed if elapsed > 0 else 0

            # OSD
            annotated = draw_osd(annotated, fps_display, len(plates), total_saved)

            # Print detections to console
            for p in plates:
                if p["plate"]:
                    print(f"  [Frame {frame_idx:>5}] Plate: {p['plate']:<14}  "
                          f"Det: {p['det_conf']:.0%}  OCR: {p['ocr_conf']:.0%}")

            if writer:
                writer.write(annotated)

            if args.show:
                cv2.imshow("Smart License Plate Detection  |  Q to quit", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:   # Q or ESC
                    print("\n[INFO] Quitting …")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"[INFO] Video saved: {out_path}")
        cv2.destroyAllWindows()


# ─── Console helpers ──────────────────────────────────────────────────────────
def _print_plates(plates: list):
    if not plates:
        print("  No license plates detected.")
        return
    print(f"\n  {'-'*55}")
    print(f"  {'#':<4} {'Plate':<15} {'Det Conf':>9}  {'OCR Conf':>9}  BBox")
    print(f"  {'-'*55}")
    for i, p in enumerate(plates, 1):
        bbox = p["bbox"]
        print(f"  {i:<4} {p['plate'] or '-':<15} "
              f"{p['det_conf']:>9.1%}  {p['ocr_conf']:>9.1%}  {bbox}")
    print(f"  {'-'*55}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Validate weights
    if not Path(args.weights).exists():
        print(f"[ERROR] Weights file not found: {args.weights}")
        sys.exit(1)

    print("=" * 65)
    print("  Smart License Plate Detection + Recognition")
    print("=" * 65)
    print(f"  Source  : {args.source}")
    print(f"  Weights : {args.weights}")
    print(f"  Conf    : {args.conf}   IoU: {args.iou}")
    print(f"  OCR     : {'Enabled (EasyOCR)' if not args.no_ocr else 'Disabled'}")
    print(f"  Database: {'Disabled' if args.no_db else args.db}")
    print("=" * 65)

    # Load YOLOv5 model
    model = load_model(args.weights, args.conf, args.iou)

    # Load OCR function (lazy)
    ocr_func = None
    if not args.no_ocr:
        try:
            from ocr_pipeline import extract_plate_text
            ocr_func = extract_plate_text
            print("[INFO] EasyOCR pipeline loaded.")
        except ImportError as e:
            print(f"[WARNING] Could not import ocr_pipeline: {e}")
            print("          Install EasyOCR with:  pip install easyocr")

    # Load DB save function
    db_save_func = None
    if not args.no_db:
        try:
            from database import save_detection, init_db
            init_db(args.db)
            db_save_func = save_detection
            print(f"[INFO] Database ready: {args.db}")
        except Exception as e:
            print(f"[WARNING] Database unavailable: {e}")

    print()

    # Route by source type
    source  = args.source.strip()
    is_file = Path(source).exists()
    is_cam  = source.isdigit()

    if is_file:
        ext = Path(source).suffix.lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            run_video_or_webcam(source, model, ocr_func, db_save_func, args)
        else:
            run_image(source, model, ocr_func, db_save_func, args)
    elif is_cam:
        run_video_or_webcam(source, model, ocr_func, db_save_func, args)
    else:
        print(f"[ERROR] Source not found or not recognised: {source}")
        print("  Use --source 0 for webcam, or provide a valid image/video path.")
        sys.exit(1)


if __name__ == "__main__":
    main()
