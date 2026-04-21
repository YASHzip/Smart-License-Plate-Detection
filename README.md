# Smart License Plate Detection

> Real-time license plate detection + OCR + database storage using YOLOv5 and EasyOCR.

---

## 📌 Project Overview

**Smart License Plate Detection** is an end-to-end machine learning pipeline that:
1. **Detects** license plates in images, videos, or live webcam feeds using a fine-tuned **YOLOv5s** model
2. **Reads** the plate text using **EasyOCR** with multi-variant preprocessing for better accuracy on small plates
3. **Stores** every detection (plate number, confidence, timestamp, source) in a **SQLite database**

### Use Cases
- Traffic monitoring & surveillance
- Parking management systems
- Toll collection automation
- Law enforcement support

---

## 🏆 Model Performance

| Metric | Training (best epoch) | Test Set |
|--------|-----------------------|----------|
| mAP@0.5 | **90.0%** | **92.3%** |
| mAP@0.5:0.95 | 43.8% | 54.1% |
| Precision | 77.7% | 88.4% |
| Recall | 85.4% | 90.0% |
| F1 Score | 81.4% | — |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Object Detection | YOLOv5s (custom fine-tuned) |
| OCR | EasyOCR |
| Database | SQLite3 |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch (CUDA-enabled) |
| Language | Python 3.8+ |
| GPU | NVIDIA (CUDA 12.x recommended) |

---

## 📁 Project Structure

```
Smart-License-Plate-Detection/
├── detect_and_recognize.py     # End-to-end pipeline (detect + OCR + save to DB)
├── ocr_pipeline.py             # EasyOCR plate text extractor (multi-variant preprocessing)
├── evaluate.py                 # Full test set evaluation (mAP, Precision, Recall, F1)
├── database.py                 # SQLite database interface
├── testing.py                  # Quick single-image inference test
├── xmltotxtconvert.py          # XML → YOLO TXT annotation converter
├── data.yaml                   # Dataset paths config for YOLOv5
├── requirements.txt            # Python dependencies
├── run_all.bat                 # One-click Windows batch script (runs all steps)
├── COMMANDS.txt                # Full command reference card
├── evaluation_report.txt       # Auto-generated model metrics report
├── plates.db                   # SQLite database (auto-created on first run)
├── plate_crop_test.jpg         # Sample plate crop for OCR testing
├── crop_251~255.jpg            # Test plate crops from detection
├── evaluation_output/          # Annotated sample images from evaluate.py
├── detection_output/           # Output from detect_and_recognize.py
├── archive/                    # Dataset (download from Kaggle)
│   ├── images/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── validation/
│       └── test/
└── yolov5/                     # YOLOv5 framework (included in repo)
    ├── models/
    │   └── experimental.py     # Patched: weights_only=False (PyTorch 2.6+ fix)
    ├── utils/
    │   └── metrics.py          # Patched: np.trapz → NumPy 2.x compat fix
    └── runs/train/exp2/
        └── weights/
            └── best.pt         # Trained model weights (download separately)
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YASHzip/Smart-License-Plate-Detection.git
cd Smart-License-Plate-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **GPU Users (Recommended):** Install PyTorch with CUDA support for significantly faster inference:
> ```bash
> pip uninstall torch torchvision -y
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
> ```
> Verify GPU is detected:
> ```bash
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
> ```

### 3. Download the Dataset
Download the dataset from Kaggle and place it in the `archive/` folder:

📎 [Kaggle Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)

Expected structure:
```
archive/
├── images/   (train / validation / test)
└── labels/   (train / validation / test)
```

### 4. Convert XML Annotations to YOLO Format
```bash
python xmltotxtconvert.py
```

### 5. Train the Model
```bash
python yolov5/train.py --img 416 --batch 4 --epochs 100 --data data.yaml --weights yolov5s.pt --device 0
```
Trained weights are saved to: `yolov5/runs/train/exp2/weights/best.pt`

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Quick inference test on a single image
python testing.py --image archive/images/test/Cars255.png

# 3. Run full model evaluation
python evaluate.py --skip-val

# 4. Run full pipeline (detection + OCR + database)
python detect_and_recognize.py --source archive/images/test/Cars255.png

# 5. Live webcam detection
python detect_and_recognize.py --source 0

# 6. View saved plate records
python database.py
```

Or just double-click **`run_all.bat`** on Windows to run all steps interactively.

---

## 🔍 Usage Guide

### Testing – Quick Inference
```bash
python testing.py                                         # default test image
python testing.py --image path/to/image.png               # custom image
python testing.py --image img.jpg --conf 0.3 --no-save    # lower confidence
```

### Evaluation – mAP, Precision, Recall, F1
```bash
python evaluate.py                   # full test-set evaluation
python evaluate.py --skip-val        # training summary only (faster)
```
Output: `evaluation_report.txt` + annotated images in `evaluation_output/`

### OCR Pipeline – Read Plate Text
```bash
# Run on a cropped plate image
python ocr_pipeline.py --image plate_crop_test.jpg

# Without preprocessing
python ocr_pipeline.py --image plate_crop_test.jpg --no-preprocess
```

### Full Pipeline – Detect + OCR + Save to DB
```bash
python detect_and_recognize.py --source archive/images/test/Cars255.png   # image
python detect_and_recognize.py --source archive/images/test/              # folder
python detect_and_recognize.py --source road_video.mp4                    # video
python detect_and_recognize.py --source 0                                 # webcam
```

**Useful flags:**

| Flag | Description |
|------|-------------|
| `--no-save` | Don't save annotated output to disk |
| `--no-db` | Skip database saving |
| `--no-ocr` | Detection bounding boxes only |
| `--conf 0.5` | Set detection confidence threshold |

### Database Management
```bash
python database.py              # view all records
python database.py --search MH  # search by plate text
python database.py --count      # total record count
python database.py --clear      # delete all records
```

**Database Schema (`plates.db → detections`):**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-incremented primary key |
| `plate_number` | TEXT | Cleaned plate text (e.g. `MH12AB1234`) |
| `raw_ocr_text` | TEXT | Raw EasyOCR output |
| `image_path` | TEXT | Source file path |
| `detection_confidence` | REAL | YOLOv5 bounding box confidence |
| `ocr_confidence` | REAL | EasyOCR average confidence |
| `source` | TEXT | `image`, `video`, or `webcam` |
| `timestamp` | TEXT | Detection datetime (`YYYY-MM-DD HH:MM:SS`) |

---

## 📋 All Commands Reference

See **`COMMANDS.txt`** for a complete copy-paste command reference, or run **`run_all.bat`** on Windows for a step-by-step guided execution.

---

## 📄 License

This project is for academic and educational purposes.  
YOLOv5 is licensed under [AGPL-3.0](https://github.com/ultralytics/yolov5/blob/master/LICENSE).
