# Smart License Plate Detection — Codebase Explanation

End-to-end pipeline for detecting and recognizing Indian license plates from images, videos, and live webcam. Uses a fine-tuned YOLOv5s model (150 epochs, 3,546 images) achieving **99.4% mAP@0.5**.

---

## 1. Data Preparation — `xmltotxtconvert.py`

Converts the Kaggle dataset from Pascal VOC XML → YOLO TXT format.
- Parses XML bounding boxes (`xmin, ymin, xmax, ymax`), normalizes to image dimensions, writes one `.txt` label per image in `class_id x_center y_center width height` format.
- Processes train / validation / test splits.

---

## 2. Indian Dataset Pipeline

### `prepare_indian_dataset.py`
Merges multiple Indian plate datasets into one unified set split 80/10/10.
- Auto-detects annotation format (YOLO TXT, Pascal VOC XML, or unlabelled).
- Deduplicates by filename hash, shuffles, copies to `indian_dataset/images` & `indian_dataset/labels`.
- Writes `indian_data.yaml` consumed by YOLOv5 training.

### `generate_synthetic_data.py`
Creates realistic synthetic training images by compositing plate crops onto background car photos.
- Applies random scale, rotation, perspective warp, brightness jitter, and motion blur.
- Auto-computes YOLO bounding box labels from the paste position.
- Output: `synthetic_dataset/` ready to be ingested by `prepare_indian_dataset.py`.

### `auto_annotate.py`
Auto-labels unlabelled images using the current best model weights.
- Runs inference, writes YOLO TXT labels for detections above `--min-conf`.
- Optionally copies source images alongside labels for direct use as a new training set.

---

## 3. Training & Fine-Tuning

Uses YOLOv5 `train.py` with transfer learning from the baseline model:

| Parameter | Value |
|-----------|-------|
| `--img` | 640 |
| `--batch` | 8 (RTX 3050 4 GB) |
| `--epochs` | 150 |
| `--weights` | `exp2/best.pt` (baseline) |
| `--hyp` | `hyp_indian.yaml` |

**`hyp_indian.yaml`** — Custom hyperparameters: higher HSV augmentation, 10° rotation, 5° shear, mixup=0.1, 5-epoch warmup, cosine LR decay to 0.001.

**Crash Recovery** — `last.pt` is saved every epoch. Resume with:
```bash
python train.py --resume runs\train\indian_finetune6\weights\last.pt
```

**`notify_when_done.ps1`** — Polls `results.csv` every 60 s; fires audio beeps + Windows toast notification when all epochs finish.

---

## 4. Evaluation — `testing.py` & `evaluate.py`

### `testing.py`
Quick single-image inference. Loads `best.pt`, prints bounding boxes, optionally saves annotated output.

### `evaluate.py`
- Reads best epoch from `results.csv` (Precision, Recall, F1, mAP).
- Runs `val.py` as subprocess on the test split for independent test-set mAP.
- Saves up to 5 annotated sample images to `evaluation_output/`.
- Writes `evaluation_report.txt`.

**Final model results (indian_finetune6, 150 epochs):**

| Metric | Score |
|--------|-------|
| mAP@0.5 | **99.4%** |
| mAP@0.5:0.95 | **89.4%** |
| Precision | **99.3%** |
| Recall | **99.7%** |
| Speed | **5.5 ms/image** |

---

## 5. OCR — `ocr_pipeline.py`

Turns a cropped plate image into text.

- **Preprocessing variants**: 3× Lanczos upscale + padding, unsharp mask sharpening, CLAHE + adaptive threshold, Otsu threshold, morphological top-hat.
- **EasyOCR** reads all variants; picks the result with highest average character confidence.
- **`clean_plate_text`**: keeps uppercase alphanumerics + hyphens; rejects obvious noise.

---

## 6. Database — `database.py`

SQLite (`plates.db`) storage for all detections.

- Table `detections`: `plate_number`, `raw_ocr_text`, `image_path`, `detection_confidence`, `ocr_confidence`, `source`, `timestamp`.
- Functions: `init_db`, `save_detection`, `get_all`, `search_plate`, `clear_all`.
- CLI: `python database.py --search MH`, `--count`, `--clear`.

---

## 7. Full Pipeline — `detect_and_recognize.py`

Main application tying all components together.

**Supported sources:**
- Single image file
- **Folder of images** (new — iterates every image in the directory)
- Video file (`.mp4`, `.avi`, `.mov`, `.mkv`)
- Live webcam (`--source 0`)

**Per-frame loop:**
1. YOLOv5 detects plate bounding boxes.
2. Crop is passed to `ocr_pipeline.py` for text extraction.
3. Annotated frame saved to `detection_output/` and logged to SQLite.

**BestFrameTracker** (video/webcam):
- Fuzzy-matches OCR readings across frames (≥80% similarity = same plate) using `difflib.SequenceMatcher`.
- Scores frames: `det_conf × ocr_conf × (plate_area / frame_area)`.
- Saves exactly **one** best-quality frame per physical plate at session end.

**Auto weight selection**: prefers `indian_finetune6/best.pt`, falls back to `exp2/best.pt`.

---

## 8. Execution Script — `run_all.bat`

Windows one-click demonstrator:
1. Installs dependencies.
2. Runs evaluation → generates report.
3. Tests OCR pipeline on a sample crop.
4. Runs full end-to-end pipeline on a test image.
5. Displays database records.
