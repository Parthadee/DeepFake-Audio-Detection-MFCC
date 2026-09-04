# 🔊 DeepFake Voice Detection

An end-to-end machine learning pipeline built to tell **real human speech apart from AI-generated (deepfake) voices**. It combines MFCC-based audio features, spectral statistics, and a mix of **Deep Learning (CNN)** and **classical ML models (Random Forest, KNN)**.

---

## ⭐ Highlights

- Classifies **real vs. fake** speech with **98%+ accuracy**
- Built on **MFCCs**, **Mel-spectrogram statistics**, and **spectral audio features**
- Implements a **1D CNN**, **Random Forest**, and **KNN** classifier
- Ships with complete evaluation metrics and plot placeholders
- Designed as a foundation for future **fraud detection and security** use cases

---

## 📂 Dataset

- **Format:** 16 kHz, mono WAV files
- **Labels:** Real / Fake
- Covers multiple **SNR levels** and **noise-reduction techniques**
- Class imbalance addressed via **SMOTE** oversampling

---

## 🎧 Feature Engineering

### **MFCC Features (40-D)**
- Extracted with Librosa
- Averaged (mean-pooled) across the time axis
- Fed into the 1D CNN

### **Engineered Features (318-D)**
- Mean MFCC values
- Mean Mel-Spectrogram values
- Mean log-STFT (spectrogram) values
- Used as input for the Random Forest and KNN models

---

## 🏗️ Architecture Overview

```
           ┌─────────────── Preprocessing ───────────────┐
           │                                               │
Audio → Load → Normalize → MFCC / Spectrogram Extraction → Features
           │                                               │
           └──────────────────────┬────────────────────────┘
                                  │
       ┌──────────────────────────┴──────────────────────────┐
       │                                                     │
  318-D Engineered Features                          MFCC Map (40×1)
       │                                                     │
Random Forest / KNN                                     1D CNN Model
       │                                                     │
       └──────────────────────────┬──────────────────────────┘
                                  ↓
                           **Real / Fake**
```

---

## 🧠 Models

### **1️⃣ 1D CNN (MFCC-Based)**

**Architecture**
- Conv1D → Dropout
- MaxPooling
- Conv1D → Dropout
- Dense → Softmax
- Trained across 40 epochs

**Input features**
![CNN Features](/assets/mfcc-40.png)

**Results**
- **Dev accuracy:** ~86%
- **Eval accuracy:** ~88%
- Performs especially well at catching **fake audio**
- Shows mild signs of overfitting

---

## 📊 CNN Evaluation (Placeholders)

### 🟦 Accuracy Over Training
![CNN Accuracy Curve](/assets/accuracy.png)

### 🟥 Loss Over Training
![CNN Loss Curve](/assets/loss.png)

### 🟩 Confusion Matrix — CNN
![CNN Confusion Matrix](/assets/confussionmatrix-cnn.png)

### 🟪 ROC Curve — CNN
![CNN ROC Curve](/assets/roc.png)

### 🟨 Precision–Recall Curve — CNN
<img width="631" height="468" alt="precision and recall" src="https://github.com/user-attachments/assets/6528d17a-c123-4064-9352-2353b8329af9" />

---

### Input features for the 318-dimensional models
![RF & KNN Features](/assets/realvsfake.png)

### **2️⃣ Random Forest Classifier**

**Results**
- **Accuracy:** **98.82%**
- Strong precision and recall across both classes
- Highly resilient to noise and variation in the dataset

**Confusion Matrix — RF**
![RF Confusion Matrix](/assets/confusion.png)

---

### **3️⃣ K-Nearest Neighbours (KNN)**

**Results**
- **Accuracy:** **98.29%**
- Consistent performance across sample variations
- Best results at **k = 7**

**Confusion Matrix — KNN**
![KNN Confusion Matrix](/assets/download.png)

---

## 📊 Model Comparison

| Model | Accuracy | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| **Random Forest** | ⭐ **98.82%** | Top performer, handles noise well | Training slows down on very large datasets |
| **KNN (k=7)** | 98.29% | Simple, yet highly competitive | Inference gets slow at scale |
| **CNN (MFCCs)** | ~88% | Captures temporal patterns | Prone to overfitting |

---

## 🚧 Known Limitations

- Class imbalance required oversampling to correct
- CNN accuracy is capped by relying on MFCCs alone
- Not yet tested against unseen/novel deepfake generation methods
- Limited testing on noisy, real-world recordings

---

## 🚀 Roadmap

- Move to **2D CNNs** trained on spectrogram images
- Integrate transformer-based encoders (**wav2vec 2.0**, **HuBERT**, **Whisper**)
- Package as a **web or mobile app** for real-time detection
- Strengthen **adversarial robustness**
- Add **explainable AI** support for forensic use cases

---

## 🛠️ Tech Stack

- **Python**
- **Librosa** — audio processing
- **TensorFlow / Keras** — CNN model
- **scikit-learn** — Random Forest, KNN, SMOTE
- **NumPy / pandas** — data preprocessing

---

## 🙌 Contributors

**Srujan Rana**
