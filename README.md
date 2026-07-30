# 🦵 IoT-Based Intelligent Physiotherapy Monitoring and Real-Time Feedback System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Soft%20Voting-green.svg)
![Platform](https://img.shields.io/badge/Platform-ESP32-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An IoT-based intelligent physiotherapy system that monitors knee rehabilitation exercises in real time using wearable motion sensors and machine learning. The system classifies rehabilitation movements and provides objective feedback to support home-based physiotherapy.

---

# 📖 Overview

Traditional physiotherapy largely depends on periodic clinical supervision, making it difficult to continuously monitor patients performing rehabilitation exercises at home. Incorrect exercise execution can reduce recovery effectiveness and increase the risk of injury.

This project addresses these challenges by combining wearable IoT sensors with machine learning to automatically recognize knee rehabilitation exercises and provide accurate movement classification in real time.

The system utilizes two MPU6050 inertial sensors connected to an ESP32 microcontroller for motion acquisition. Sensor data is processed using signal preprocessing techniques before being classified using a Soft Voting ensemble learning model.

---

# ✨ Features

- Real-time knee rehabilitation monitoring
- Wearable dual MPU6050 sensor setup
- ESP32-based wireless data acquisition
- Signal preprocessing using complementary filtering
- Sliding window feature extraction
- Soft Voting ensemble classifier
- High classification accuracy for rehabilitation exercises
- Designed for home-based physiotherapy assistance

---

# 🏗️ System Architecture

> <img width="488" height="266" alt="image" src="https://github.com/user-attachments/assets/e5cd7851-d63d-406e-854c-558d29e1dc06" />

---

# 🛠️ Hardware

- ESP32 Development Board
- MPU6050 IMU Sensor ×2
- Wearable Knee Mount
- Rechargeable Power Supply

---

# 💻 Software

- Python
- Arduino IDE
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Joblib

---

# 🤖 Machine Learning Pipeline

The rehabilitation movement recognition pipeline consists of:

1. Sensor Data Collection
2. Complementary Filter
3. Signal Cleaning
4. Sliding Window Segmentation
5. Feature Extraction
6. Feature Scaling
7. Soft Voting Ensemble Classification
8. Exercise Prediction

---

# 📂 Project Structure

```
IoT-Intelligent-Physiotherapy-Monitoring/
│
├── config/
├── data/
├── docs/
├── models/
│   ├── knee_model.pkl
│   └── knee_scaler.pkl
│
├── notebooks/
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── classification_report.csv
│   └── ...
│
├── src/
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 📊 Results

The Soft Voting ensemble model achieved strong classification performance on the rehabilitation exercise dataset.

| Metric | Value |
|---------|--------|
| Accuracy | **92.74%** |
| Precision | **0.92** |
| Recall | **0.94** |
| AUC Score | **0.971** |

---

## Confusion Matrix

> *(Insert `results/confusion_matrix.png` here.)*

```markdown
![Confusion Matrix](results/confusion_matrix.png)
```

---

## ROC Curve

> *(Insert your ROC curve image here.)*

```markdown
![ROC Curve](results/roc_curve.png)
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Sudeep70/IoT-Intelligent-Physiotherapy-Monitoring.git
```

Navigate into the project directory:

```bash
cd IoT-Intelligent-Physiotherapy-Monitoring
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Train the model:

```bash
python src/train.py
```

Run inference:

```bash
python src/predict.py
```

If using the pretrained model:

```python
import joblib

model = joblib.load("models/knee_model.pkl")
scaler = joblib.load("models/knee_scaler.pkl")
```

---

# 📈 Dataset

The dataset used for training is **not included** in this repository because of its size.

Sensor data consists of motion readings collected from dual MPU6050 inertial sensors during knee rehabilitation exercises.

---

# 🔮 Future Work

- Mobile application integration
- Cloud-based patient monitoring
- Personalized rehabilitation recommendations
- Expanded rehabilitation exercise library
- Clinical validation with larger participant groups

---

# 👨‍💻 Authors

**Sudeep**

Department of Computer Science and Engineering

---

# 📄 Citation

If you use this work in your research, please cite the associated IEEE conference paper.

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Acknowledgements

This project was developed as part of an undergraduate engineering project focused on applying IoT and machine learning to intelligent physiotherapy monitoring and rehabilitation assistance.
