# 🎭 Facial Emotion Recognition & Affective Analytics Studio

A production-grade Deep Learning system engineered with **PyTorch**, **OpenCV**, and **Streamlit** for real-time facial expression classification, affective sentiment analysis (Russell's Circumplex Model), and **Grad-CAM visual explainability**.

Trained on **FER-2013** across 7 fundamental emotion classes: *Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise*.

---

## 🌟 Key Features

- **Deep VGG-Style CNN Backbone**: Custom 8-convolutional-layer architecture with Batch Normalization, Dropout (0.25 / 0.50), and Adaptive Average Pooling (~1.25M parameters, sub-5 MB binary).
- **Grad-CAM Visual Saliency**: Real-time Gradient-weighted Class Activation Mapping highlighting the salient facial regions (eyes, mouth, brow) guiding predictions.
- **Affective Valence & Arousal Analytics**: Continuous 2D circumplex modeling tracking emotional polarity (+1.0 to -1.0) and expressive intensity.
- **Dynamic Prior Calibration**: Integrated sensitivity tuning for challenging micro-expressions (boosts Fear recall from 9.6% to 21.1% without accuracy trade-offs).
- **Interactive Vision Studio**: Modular Streamlit application with live webcam ingestion, drag-and-drop file support (`.jpg`, `.png`, `.jfif`, `.webp`), session telemetry, and CSV export.
- **Fast CPU & GPU Inference**: Sub-28 ms end-to-end latency on standard CPUs (~35 FPS).

---

## 🏗️ Neural Architecture

```
Input (48×48 Grayscale)
  │
  ├── Block 1: [Conv2D 3×3 (32)] × 2 ➔ BatchNorm ➔ ReLU ➔ MaxPool(2) ➔ Drop2D(0.25)
  ├── Block 2: [Conv2D 3×3 (64)] × 2 ➔ BatchNorm ➔ ReLU ➔ MaxPool(2) ➔ Drop2D(0.25)
  ├── Block 3: [Conv2D 3×3 (128)] × 2 ➔ BatchNorm ➔ ReLU ➔ MaxPool(2) ➔ Drop2D(0.25)
  └── Block 4: [Conv2D 3×3 (256)] × 2 ➔ BatchNorm ➔ ReLU ➔ MaxPool(2) ➔ Drop2D(0.25)
  │
AdaptiveAvgPool2D(1×1) ➔ Flatten (256-D)
  │
Linear(256 ➔ 256) ➔ ReLU ➔ Dropout(0.50) ➔ Linear(256 ➔ 7) ➔ Softmax
```

---

## 📊 Benchmark Results (FER-2013 Held-Out Test Set)

Evaluated across **7,178** untouched test portraits:

| Emotion Class | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Happy** | **80.5%** | **85.5%** | **0.829** | 🟢 High discrimination |
| **Surprise** | **68.0%** | **81.9%** | **0.743** | 🟢 Distinct mouth & eye opening |
| **Neutral** | 50.2% | 65.8% | 0.570 | 🟡 Baseline resting face |
| **Disgust** | 35.7% | 45.9% | 0.402 | 🟠 Compensated via class weights |
| **Angry** | 47.6% | 39.2% | 0.430 | 🟡 Brow furrow features |
| **Sad** | 44.0% | 38.1% | 0.408 | 🟡 Subtle lip depression |
| **Fear** | 32.4% | 21.1% | 0.255 | 🔴 Improved via sensitivity prior |
| **Overall Accuracy** | — | — | **57.48%** | *Human agreement on FER-2013 is ~65%* |
| **Macro F1** | — | — | **52.00%** | Unweighted mean across all classes |

---

## 📁 Repository Structure

```
facial_emotion_recognition/
├── .gitignore                     # Ignores virtualenvs, dataset caches, logs
├── README.md                      # Primary project documentation
├── emotion-dl/                    # Application & deep learning codebase
│   ├── app.py                     # Streamlit Vision Studio Web App
│   ├── model.py                   # EmotionCNN, Grad-CAM, & face detection pipeline
│   ├── train.py                   # Training loop with Focal Loss & OneCycleLR
│   ├── evaluate.py                # Test set evaluation & confusion matrix exporter
│   ├── realtime.py                # Lightweight OpenCV live webcam loop
│   ├── requirements.txt           # Environment dependencies
│   └── checkpoints/               # Trained weights & diagnostic plots
│       ├── best.pt                # Trained model weights (4.98 MB)
│       ├── confusion_matrix.png   # Normalized test confusion matrix
│       ├── training_curves.png    # Loss & accuracy trajectories
│       └── metrics_summary.json   # Machine-readable evaluation report
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Cosmo29/facial_emotion_recognition.git
cd facial_emotion_recognition/emotion-dl

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Streamlit Studio
```bash
streamlit run app.py
```
*Access the interactive interface at `http://localhost:8501` to test with web camera snapshots or image uploads (`.jpg`, `.jpeg`, `.png`, `.jfif`, `.webp`).*

### 3. Run Live Webcam Detection (OpenCV)
```bash
python realtime.py --camera 0
```
*Press `q` to exit.*

### 4. Evaluate Test Metrics
```bash
python evaluate.py --fear-boost 1.4
```

---

## 📜 License
MIT License. Developed for research and educational purposes.
