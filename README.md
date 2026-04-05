# 🌸 Iris Flower Classifier

> A machine learning web app that predicts iris species from petal & sepal measurements using Random Forest, with real-time confidence scores and model insights.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?style=flat-square&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen?style=flat-square)

---

## 📌 Overview

The **Iris Flower Classifier** is a Flask-based web application that uses a **Random Forest** machine learning model trained on the full Iris dataset (150 samples) to classify iris flowers into three species:

- 🌼 **Setosa**
- 💜 **Versicolor**
- 🌺 **Virginica**

---

## ✨ Features

- ✅ Trained on full 150-sample Iris dataset
- 📊 Confusion Matrix with heatmap visualization
- 📋 Classification Report (Precision, Recall, F1-Score)
- 🌿 Feature Importance chart & ranked bar visualization
- 🔍 Real-time flower species prediction with confidence scores
- ⚠️ Input validation with helpful error messages
- 🎨 Beautiful dark-themed responsive UI

---

## 🗂️ Project Structure

```
iris_web/
├── app.py                  # Flask backend & ML model
├── requirements.txt        # Python dependencies
└── templates/
    └── index.html          # Frontend UI
```

---

## ⚙️ How It Works

```
User Input (measurements)
        ↓
Flask Backend (app.py)
        ↓
Random Forest Model
        ↓
Predicted Species + Confidence %
```

The model is trained once at server startup using `scikit-learn`'s built-in Iris dataset. Input measurements are validated, passed to the model, and predictions are returned as JSON to the frontend.

---

## 🚀 How to Run

### 1. Clone or download the project

```bash
git clone https://github.com/yourusername/CodeAlpha_IrisFlowerClassification.git
cd iris-classifier/iris_web
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

### 5. Open in browser

```
http://localhost:5000
```

> ℹ️ You will see `Running on http://127.0.0.1:5000` in your terminal — this is normal. The app is live!

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web framework |
| `scikit-learn` | Random Forest model & evaluation |
| `pandas` | Data handling |
| `numpy` | Numerical operations |
| `matplotlib` | Chart generation |
| `seaborn` | Confusion matrix heatmap |

Install all at once:

```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn
```

---

## 🌸 Input Ranges

When predicting a new flower, use these measurement ranges:

| Feature | Min | Max | Example |
|---|---|---|---|
| Sepal Length (cm) | 4.0 | 8.5 | 5.1 |
| Sepal Width (cm) | 1.5 | 5.0 | 3.5 |
| Petal Length (cm) | 0.5 | 7.5 | 1.4 |
| Petal Width (cm) | 0.1 | 3.0 | 0.2 |

> 💡 The above example values will predict **Setosa**

---

## 📊 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Number of Trees | 100 |
| Train / Test Split | 80% / 20% (stratified) |
| Dataset Size | 150 samples |
| Classes | setosa, versicolor, virginica |
| Typical Accuracy | 95% – 100% |

---

## 🖥️ Screenshots

| Section | Description |
|---|---|
| Hero + Stats | Accuracy, sample counts, tree count |
| Model Evaluation | Confusion matrix + classification report |
| Feature Importance | Chart + animated importance bars |
| Predict | Input form + live species prediction |

---

## 📄 License

This project is for educational purposes. The Iris dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

---

## 👤 Author

Built with ❤️ using Python, Flask, and Scikit-learn.
