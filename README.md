# SmartLicensePlateDetection

## 🚗 **Project Overview**  
SmartLicensePlateDetection is a machine learning-based solution designed for real-time detection and recognition of vehicle license plates from images or video streams. The system utilizes **YOLOv5 (You Only Look Once)** for efficient object detection and integrates **Optical Character Recognition (OCR)** via EasyOCR to extract text from license plates. The extracted plate numbers are automatically saved into a **SQLite database** for easy retrieval and further processing.

This solution can be used in various applications including:
- Traffic monitoring systems
- Parking management
- Toll collection
- Law enforcement and surveillance

---

## 🛠️ **Features**
- **Accurate License Plate Detection:** Identifies and locates license plates in various lighting and weather conditions.
- **Optical Character Recognition (OCR):** Extracts text from detected license plates using EasyOCR.
- **Real-Time Processing:** Supports image, video file, and live webcam input.
- **Customizable Training:** Train the detection model with your own dataset for region-specific license plate recognition.
- **Database Integration:** Automatically stores detected license plate numbers, confidence scores, timestamps, and source info in SQLite.
- **Cross-Platform Compatibility:** Compatible with Windows, Linux, and cloud-based environments.

---

## 🔧 **Installation Guide**
To run this project, you'll need the following dependencies:

- torch
- torchvision
- numpy
- opencv-python
- easyocr
- matplotlib
- Pillow

You can install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### **1. Clone the Repository**
```bash
git clone https://github.com/YASHzip/Smart-License-Plate-Detection.git
cd Smart-License-Plate-Detection
```

### **2. YOLOv5 installation**
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -U -r requirements.txt
cd ..
```

### **3. Prepare the Dataset**
Organize your dataset in following order:
```
archive/
├── annotations/
│   ├── test/
│   ├── train/
│   └── validation/
├── images/
│   ├── test/
│   ├── train/
│   └── validation/
```
Dataset source: [Kaggle Car Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?select=images)

### **4. Convert XML files to txt**
Convert the XML annotation files to YOLO `.txt` format. Update the paths inside `xmltotxtconvert.py` to match your system:
```bash
python xmltotxtconvert.py
```

### **5. Training the model**
Train your YOLOv5 model with your custom dataset by opening a terminal inside the `yolov5/` directory:
```bash
python train.py --img 416 --batch 4 --epochs 100 --data ../data.yaml --weights yolov5s.pt --device 0
```
Trained weights are saved at:  
`yolov5/runs/train/exp2/weights/best.pt`

---

### **6. 🧪 Testing and Evaluation**

#### Quick Inference Test
Test the trained model on a single image:
```bash
python testing.py                                        # uses default test image
python testing.py --image path/to/image.png              # custom image
python testing.py --image img.jpg --conf 0.3 --no-save   # lower confidence threshold
```

#### Evaluation on Test Set (mAP, Precision, Recall, F1)
Run a full evaluation on the test dataset:
```bash
python evaluate.py
python evaluate.py --weights yolov5/runs/train/exp2/weights/best.pt --img 416
python evaluate.py --skip-val    # only show training results summary
```
The evaluation report is saved to `evaluation_report.txt` and annotated sample images to `evaluation_output/`.

#### 📊 Training Results (100 Epochs)
| Metric | Best Value |
|--------|-----------|
| mAP@0.5 | **~90.0%** |
| mAP@0.5:0.95 | **~52.6%** |
| Precision | **~86%** |
| Recall | **~85%** |

---

### **7. 🔍 Full Detection + OCR Pipeline**
Run the end-to-end pipeline (detection + plate text recognition + database storage):

```bash
# Single image
python detect_and_recognize.py --source Cars252.png

# Video file
python detect_and_recognize.py --source road_video.mp4

# Live webcam (index 0)
python detect_and_recognize.py --source 0

# Disable OCR (detection only)
python detect_and_recognize.py --source 0 --no-ocr

# Skip database saving
python detect_and_recognize.py --source image.jpg --no-db
```

**Output:** Annotated image/video saved to `detection_output/`. Press **Q** or **ESC** to quit the webcam/video window.

---

### **8. 🗄️ Database Management**
All detected plates are automatically saved to `plates.db` (SQLite).

```bash
# View all stored detections
python database.py

# Search for a specific plate number
python database.py --search MH12

# Count total records
python database.py --count

# Clear all records
python database.py --clear
```

**Database Schema (`plates.db` → `detections` table):**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-incremented primary key |
| `plate_number` | TEXT | Cleaned plate text (e.g. MH12AB1234) |
| `raw_ocr_text` | TEXT | Raw EasyOCR output |
| `image_path` | TEXT | Source image/video path |
| `detection_confidence` | REAL | YOLOv5 bounding box confidence |
| `ocr_confidence` | REAL | EasyOCR average confidence |
| `source` | TEXT | `image`, `video`, or `webcam` |
| `timestamp` | TEXT | Detection datetime (YYYY-MM-DD HH:MM:SS) |

---

## 📁 **Project Structure**
```
Smart-License-Plate-Detection/
├── data.yaml                   # Dataset config for YOLOv5
├── requirements.txt            # Python dependencies
├── xmltotxtconvert.py          # XML → YOLO TXT annotation converter
├── testing.py                  # Quick single-image inference test
├── evaluate.py                 # Full test set evaluation (mAP / F1)
├── ocr_pipeline.py             # EasyOCR-based plate text extractor
├── database.py                 # SQLite database interface
├── detect_and_recognize.py     # End-to-end pipeline (image/video/webcam)
├── Link to dataset.txt         # Kaggle dataset link
├── Results/                    # Training result plots
└── yolov5/                     # YOLOv5 framework (clone separately)
    └── runs/train/exp2/
        ├── weights/
        │   ├── best.pt         # Best model checkpoint
        │   └── last.pt         # Last epoch checkpoint
        └── results.csv         # Per-epoch training metrics
```

---

## 📦 **Tech Stack**
| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv5s (custom fine-tuned) |
| OCR | EasyOCR |
| Database | SQLite3 (built-in Python) |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |
| Language | Python 3.8+ |

---

## 🚀 **Quick Start (All-in-One)**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the model on an image
python testing.py --image archive/images/test/Cars255.png

# 3. Run full evaluation
python evaluate.py --skip-val

# 4. Run real-time webcam detection + OCR
python detect_and_recognize.py --source 0

# 5. View database records
python database.py
```
