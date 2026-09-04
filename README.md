<!-- <details> <summary>Click to expand</summary> -->

# Deepfake Audio Detection via MFCC Features and VGG16 + LSTM

This project implements the research paper:  
**"Deepfake Audio Detection via MFCC Features Using Machine Learning"**

---

## 📌 Summary

We detect audio deepfakes using classical ML models and a hybrid deep learning model. The project covers:

- 🧹 Data Cleaning & Preprocessing
- 🎵 MFCC and Spectral Feature Extraction
- ⚙️ Dimensionality Reduction via PCA
- 🧠 ML Models: SVM, RF, MLP, Gradient Boosting
- 🤖 Deep Learning: VGG16 + LSTM Fusion
- 🔍 Hyperparameter tuning with RandomizedSearchCV

---

## 📂 Dataset Used

**Fake-or-Real (FoR)**:  
Includes 4 subsets:
- `for-original`
- `for-2sec`
- `for-norm`
- `for-rerec`

---

## 📈 Results & Observations

- PCA helped reduce training time without losing much accuracy.
- VGG16+LSTM gave best performance for robust detection.
- Evaluation used accuracy, confusion matrix, ROC-AUC.

---

## 🔗 Notebook

Full code in this notebook:
`deepfake_audio_detection.ipynb`

Also on [Kaggle](https://www.kaggle.com/code/gixo95/deepfake-audio-detection-code)

---

## 🚀 Tech Stack

- Python, NumPy, Pandas
- Librosa, Matplotlib
- Scikit-learn
- TensorFlow / Keras

---

## 📦 How to Run

```bash
git clone https://github.com/<your-username>/deepfake-audio-detection.git
cd deepfake-audio-detection
jupyter notebook deepfake_audio_detection.ipynb
```

## 📊 Results (MLP Classifier on FoR-Rerecorded Subset)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### ROC Curve
![ROC Curve](results/roc_curve.png)
