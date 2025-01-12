# SmartLicensePlateDetection

## 🚗 **Project Overview**  
SmartLicensePlateDetection is a machine learning-based solution designed for real-time detection and recognition of vehicle license plates from images or video streams. The system utilizes **YOLOv5 (You Only Look Once)** for efficient object detection and integrates **Optical Character Recognition (OCR)** to extract text from license plates. The extracted plate numbers are automatically saved into a database for easy retrieval and further processing.

This solution can be used in various applications including:
- Traffic monitoring systems
- Parking management
- Toll collection
- Law enforcement and surveillance

---

## 🛠️ **Features**
- **Accurate License Plate Detection:** Identifies and locates license plates in various lighting and weather conditions.
- **Optical Character Recognition (OCR):** Extracts text from detected license plates.
- **Real-Time Processing:** Supports both image and video input for live or batch detection.
- **Customizable Training:** Train the detection model with your own dataset for region-specific license plate recognition.
- **Database Integration:** Automatically stores detected license plate numbers.
- **Cross-Platform Compatibility:** Compatible with Windows, Linux, and cloud-based environments.

---

## 🔧 **Installation Guide**
To run this project, you'll need the following dependencies:

- TensorFlow
- torch
- torchvision
- numpy
- opencv-python
- matplotlib
- pytesseract
- Pillow

You can install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### **1. Clone the Repository**
```bash
git clone https://github.com/YASHzip/SmartLicensePlateDetection.git
cd SmartLicensePlateDetection
```

### **2. YOLOv5 installation**
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -U -r requirements.txt
```

### **3. Prepare the Dataset**
Organize your dataset in following order:
```bash
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
### **4. Convert XML files to txt**
You'll have to covert the XML file present in the annotations directory to .txt and save them in labels directory inside the archive directory for the YOLOv5 model to train them.
To do so open and run the xmltotxtconvert.py file and change the path of the directories according to your device.

### **5. Training the model**
