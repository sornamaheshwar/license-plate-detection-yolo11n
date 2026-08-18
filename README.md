# License Plate Detection using YOLO11n

A domain-specific computer vision system that fine-tunes a pretrained **YOLO11n** object detection model to detect vehicle license plates and integrates the detector with an OCR pipeline for extracting candidate plate text.

The project demonstrates an end-to-end machine learning workflow covering **transfer learning, dataset preparation, GPU-based training, model evaluation, inference, OCR integration, and interactive deployment with Streamlit**.

---

## 🚀 Key Results

The fine-tuned YOLO11n model was evaluated on a held-out test set containing **2,048 images and 2,134 annotated license plate instances**.

| Metric | Result |
|---|---:|
| **mAP@50** | **98.93%** |
| **mAP@50–95** | **70.83%** |
| **Precision** | **98.80%** |
| **Recall** | **97.14%** |

The model was trained locally using an **NVIDIA GeForce GTX 1650 with 4 GB VRAM**.

---

## 📌 Project Overview

General-purpose object detection models such as YOLO models pretrained on the COCO dataset can recognize common objects including:

- Cars
- Buses
- People
- Motorcycles

However, `License_Plate` is not a class in the general-purpose COCO detection task.

This project adapts a pretrained **YOLO11n** model to the license plate domain through fine-tuning on a dedicated license plate dataset.

The resulting system can:

- Detect vehicle license plates
- Draw bounding boxes around detected plates
- Report detection confidence
- Crop detected license plates
- Pass plate crops to an OCR pipeline
- Generate candidate plate text
- Compare pretrained and fine-tuned YOLO11n
- Evaluate the fine-tuned model on a held-out test set
- Provide an interactive Streamlit interface

---

## 🎯 Problem Statement

The objective is to build a lightweight, domain-specific object detection system capable of identifying vehicle license plates from images.

Rather than training an object detector completely from scratch, the project uses **transfer learning**:

```text
Pretrained YOLO11n
        │
        ▼
License Plate Dataset
        │
        ▼
Fine-Tuning
        │
        ▼
Domain-Specific YOLO11n
        │
        ▼
License Plate Detection
        │
        ▼
Plate Crop
        │
        ▼
OCR
        │
        ▼
Candidate Plate Text
```

This allows the pretrained model to retain useful visual features while adapting its detection capability to a specialized domain.

---

## 🧠 Why Fine-Tuning?

The original YOLO11n model is a general-purpose object detector.

Its pretrained knowledge is useful for identifying objects such as cars, buses, and people, but it does not have a dedicated `License_Plate` class.

Fine-tuning adapts the model from:

```text
General-purpose object detection
              │
              ▼
        YOLO11n / COCO
              │
              ▼
      Fine-tuning on
   license plate annotations
              │
              ▼
 Specialized license plate
       object detector
```

The key objective is therefore **domain adaptation**, rather than simply training another object detector.

---

# 🏗️ System Architecture

The deployed prototype follows this pipeline:

```text
                         ┌─────────────────┐
                         │   Input Image   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    YOLO11n      │
                         │   Fine-Tuned    │
                         └────────┬────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ License Plate Detection│
                     └────────────┬───────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Plate Crop    │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     EasyOCR     │
                         │  OCR Component  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Candidate Text  │
                         └─────────────────┘
```

The application also includes a separate baseline comparison:

```text
                         Input Image
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
        Pretrained YOLO11n        Fine-Tuned YOLO11n
                 │                         │
                 ▼                         ▼
        General COCO Objects        License Plate
        Car / Bus / Person            Detection
```

---

# 📊 Dataset

The project uses a license plate detection dataset exported in **YOLO11 format**.

The dataset contains:

- Training images
- Validation images
- Test images
- YOLO-format bounding-box annotations
- One object class

### Dataset Class

```yaml
nc: 1
names: ['License_Plate']
```

The final held-out test set used for evaluation contained:

- **2,048 images**
- **2,134 annotated license plate instances**

The training dataset contained **7,058 training images**.

The complete dataset is not included in this repository because of its size.

---

# 🔬 Model Training

## Base Model

```text
YOLO11n
```

A pretrained YOLO11n model was used as the starting point for transfer learning.

## Fine-Tuned Model

```text
models/license_plate_yolo11n.pt
```

The fine-tuned model specializes in detecting:

```text
License_Plate
```

## Training Hardware

Training was performed locally using:

```text
GPU: NVIDIA GeForce GTX 1650
VRAM: 4 GB
```

GPU acceleration was enabled using PyTorch and CUDA.

---

# 📈 Model Evaluation

The fine-tuned model was evaluated on a held-out test set.

## Results

| Metric | Result |
|---|---:|
| mAP@50 | **98.93%** |
| mAP@50–95 | **70.83%** |
| Precision | **98.80%** |
| Recall | **97.14%** |

### What the metrics mean

**Precision — 98.80%**

The model produced a very low proportion of false-positive license plate detections on the evaluation set.

**Recall — 97.14%**

The model detected the large majority of annotated license plates in the test set.

**mAP@50 — 98.93%**

Measures mean average precision using an IoU threshold of 0.50.

**mAP@50–95 — 70.83%**

Evaluates detection performance across multiple, stricter IoU thresholds and therefore provides a more demanding measure of bounding-box localization quality.

---

# 🆚 Pretrained vs Fine-Tuned YOLO11n

A key feature of the application is a visual comparison between the original pretrained YOLO11n model and the fine-tuned model.

## Pretrained YOLO11n

The pretrained model is a general-purpose COCO detector and can recognize objects such as:

```text
Car
Bus
Person
Motorcycle
```

However:

```text
License_Plate
```

is not one of its detection classes.

## Fine-Tuned YOLO11n

After fine-tuning, the model becomes specialized for:

```text
License_Plate
```

The application runs both models on the same image and displays the results side-by-side.

### Concept

```text
              PRETRAINED YOLO11n

                    YOLO11n
                       │
                       ▼
              General-purpose
               object detection
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
            Car       Bus      Person


                    │
                    │ Fine-Tuning
                    ▼


              FINE-TUNED YOLO11n

                    YOLO11n
                       │
                       ▼
              License Plate
                 Detection
                       │
                       ▼
                License_Plate
```

This demonstrates how transfer learning can adapt a general-purpose detector to a domain-specific object detection task.

---

# 🔤 OCR Integration

License plate detection and text recognition are treated as **two separate stages**.

After YOLO detects a license plate:

```text
Full Image
    │
    ▼
License Plate Bounding Box
    │
    ▼
Plate Crop
    │
    ▼
OCR
    │
    ▼
Candidate Text
```

The prototype uses **EasyOCR** as a downstream OCR component.

The application displays:

- Detected plate crop
- YOLO detection confidence
- OCR candidate text
- OCR confidence

## ⚠️ OCR Disclaimer

OCR is currently an **experimental downstream component**.

The displayed text is an OCR candidate and has not been independently validated against ground-truth license plate strings.

Therefore:

> **OCR confidence should not be interpreted as OCR accuracy.**

The application intentionally communicates this limitation and recommends visually verifying the extracted text against the detected plate.

This distinction keeps the model evaluation focused on the component that was actually trained and quantitatively evaluated: **license plate detection**.

---

# 🖥️ Streamlit Application

The project includes an interactive Streamlit interface for demonstrating the complete pipeline.

The application contains three main sections.

## 1. 🔍 Live Detection

Users can provide a vehicle image to the fine-tuned model.

The application displays:

- Detected license plates
- Bounding boxes
- Detection confidence
- Cropped plate images
- OCR candidate text
- OCR confidence

---

## 2. ⚖️ Pretrained vs Fine-Tuned

The same input image is processed using:

```text
Pretrained YOLO11n
```

and:

```text
Fine-Tuned YOLO11n
```

The results are displayed side-by-side to demonstrate the effect of domain-specific fine-tuning.

The pretrained model shows general-purpose COCO detections, while the fine-tuned model detects the specialized `License_Plate` class.

---

## 3. 📊 Model Evaluation

The application displays the final test-set metrics:

```text
mAP@50       98.93%
mAP@50–95    70.83%
Precision    98.80%
Recall       97.14%
```

It also provides a short explanation of the metrics and the purpose of fine-tuning.

---

# 📁 Project Structure

```text
license-plate-detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── license_plate_yolo11n.pt
│
├── train.py
├── baseline.py
├── evaluation.py
├── inference.py
├── ocr_compare.py
│
└── yolo11n.pt
```

## Important Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `train.py` | YOLO11n fine-tuning pipeline |
| `baseline.py` | Baseline inference using pretrained YOLO11n |
| `evaluation.py` | Model evaluation |
| `inference.py` | Fine-tuned model inference |
| `ocr_compare.py` | OCR experimentation and comparison |
| `models/license_plate_yolo11n.pt` | Final fine-tuned YOLO11n model |
| `yolo11n.pt` | Original pretrained YOLO11n model |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes datasets, training artifacts, environments, and other local files |

---

# 🛠️ Technologies Used

## Machine Learning

- YOLO11n
- Transfer Learning
- Object Detection
- PyTorch
- CUDA

## Computer Vision

- OpenCV
- Image preprocessing
- Bounding-box detection
- Image cropping

## OCR

- EasyOCR
- Tesseract OCR experimentation

## Application

- Streamlit

## Programming

- Python

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd license-plate-detection
```

> Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the URL of the GitHub repository.

## 2. Create a virtual environment

```bash
python -m venv .venv
```

## 3. Activate the environment

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

Upload a vehicle image and use the available sections to:

1. Detect license plates
2. Compare pretrained and fine-tuned YOLO11n
3. View OCR candidate text
4. Review model evaluation metrics

---

# 🧪 Running Individual Components

## Train the Model

```bash
python train.py
```

Training requires the license plate dataset and its YOLO configuration file.

## Run Inference

```bash
python inference.py
```

## Evaluate the Model

```bash
python evaluation.py
```

## Run OCR Experiments

```bash
python ocr_compare.py
```

---

# ⚠️ Limitations

This project is a prototype focused primarily on **license plate detection**, rather than a production-ready Automatic Number Plate Recognition (ANPR) system.

## 1. OCR is not fully validated

The OCR component generates candidate text but does not currently have an independently evaluated OCR accuracy metric.

## 2. Dataset limitations

Model performance depends on the distribution of:

- Image quality
- Vehicle types
- Viewpoints
- Lighting conditions
- Plate formats
- Geographic characteristics

represented in the training dataset.

## 3. Real-world conditions

Detection performance may vary under:

- Motion blur
- Severe lighting changes
- Low-resolution images
- Occlusion
- Extreme viewing angles
- Damaged or partially visible plates
- Unusual plate formats

## 4. No multi-object tracking

The current prototype performs image-based detection and does not implement tracking across video frames.

## 5. No production ANPR backend

The application does not currently connect detected plate numbers to:

- Vehicle databases
- Registration systems
- External identification services
- Persistent databases

---

# 🔮 Future Improvements

The current prototype provides the core detection and OCR pipeline. Possible extensions include:

## Improved OCR

- Build a dedicated license plate OCR pipeline
- Add plate-specific preprocessing
- Apply perspective correction
- Evaluate OCR against ground-truth plate strings
- Improve character segmentation
- Introduce confidence-based candidate selection

## Video Detection

Extend the system from static images to real-time video:

```text
Webcam / Video
      │
      ▼
YOLO11n
      │
      ▼
License Plate Detection
      │
      ▼
Object Tracking
      │
      ▼
OCR
      │
      ▼
Plate Identification
```

## Object Tracking

Integrate tracking algorithms such as:

- ByteTrack
- BoT-SORT

This would allow individual vehicles and license plates to be tracked across video frames.

## Improved Localization

Potential improvements include:

- Additional training data
- More diverse samples
- Hyperparameter tuning
- Higher-resolution training
- Dataset augmentation

## Production Deployment

The prototype could be extended into a production service using:

- REST API
- Docker
- Cloud-hosted inference
- GPU inference servers
- Edge-device deployment

---

# 💡 What This Project Demonstrates

This project demonstrates practical experience across the machine learning lifecycle:

- Dataset preparation
- YOLO annotation format
- Transfer learning
- Model fine-tuning
- GPU-based training
- Object detection
- Model evaluation
- Quantitative performance analysis
- Computer vision inference
- OCR integration
- Baseline comparison
- Interactive application development
- Deployment-oriented project structuring

Rather than stopping after training a model and reporting a metric, the project connects the trained model to an interactive application and demonstrates how a general-purpose model can be adapted to a specific business domain.

---

# 📌 Key Takeaway

The central idea of this project is:

> **A general-purpose pretrained object detector can be adapted into a domain-specific detector through transfer learning and fine-tuning.**

The resulting YOLO11n model achieved:

```text
mAP@50       98.93%
mAP@50–95    70.83%
Precision    98.80%
Recall       97.14%
```

on the held-out test set.

The Streamlit application then exposes the model through an interactive interface and demonstrates the complete pipeline:

```text
Image
  ↓
YOLO11n
  ↓
License Plate Detection
  ↓
Plate Crop
  ↓
EasyOCR
  ↓
Candidate Plate Text
```

---

# 👨‍💻 Author

**M Sakthi Sorna Maheswar**

Electronics and Communication Engineering graduate focused on software development, machine learning, computer vision, and AI/ML applications.

---

# 📄 Dataset & License

The dataset used in this project is subject to its original dataset license and attribution requirements.

The dataset metadata identifies the source dataset as licensed under **CC BY 4.0**.

The complete dataset is not included in this repository.

The trained model and application code are provided for educational and portfolio demonstration purposes.