# 🎬 CinePredict AI - Movie Revenue Prediction using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg">
  <img src="https://img.shields.io/badge/Framework-FastAPI-green.svg">
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-red.svg">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

---

# 🎯 Overview

CinePredict AI is an end-to-end Data Science and Machine Learning project that predicts a movie's box office revenue before its release using historical movie data.

The platform combines data analytics, feature engineering, machine learning, explainable AI, and interactive dashboards to help production companies, investors, and analysts estimate the commercial success of upcoming movies.

---

# 🚀 Features

### 🎬 Revenue Prediction
- Worldwide Revenue Prediction
- Domestic Revenue Prediction
- Opening Weekend Prediction

### 📊 Analytics Dashboard
- Revenue Trends
- Budget Analysis
- Genre Analysis
- Director Performance
- Cast Performance
- Production Company Analysis

### 🤖 Machine Learning
- Regression Models
- Classification Models
- Model Comparison
- Hyperparameter Tuning

### 🧠 Explainable AI
- SHAP Feature Importance
- LIME Explanations
- Prediction Confidence

### 🎯 Movie Success Prediction
- Hit
- Average
- Flop

### 💡 Recommendation Engine
- Best Release Month
- Best Budget Range
- Similar Movies
- Genre Recommendation

---

# 🏗️ System Architecture

```
                 Movie Dataset
                       │
             Data Collection
                       │
               Data Cleaning
                       │
          Feature Engineering
                       │
             Machine Learning
                       │
        Revenue Prediction Model
                       │
        Explainable AI (SHAP)
                       │
        FastAPI Prediction API
                       │
      Streamlit Dashboard
                       │
              End Users
```

---

# 🛠️ Tech Stack

## Programming

- Python

## Data Analysis

- Pandas
- NumPy

## Visualization

- Matplotlib
- Plotly
- Streamlit

## Machine Learning

- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost

## Explainable AI

- SHAP
- LIME

## Backend

- FastAPI

## Database

- PostgreSQL
- MongoDB

## Deployment

- Docker
- Render
- Streamlit Cloud

---

# 📂 Project Structure

```
CinePredict-AI/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── explain.py
│   └── utils.py
│
├── models/
│
├── dashboard/
│
├── api/
│
├── reports/
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

---

# 📊 Dataset

The project uses movie datasets containing:

- Movie Title
- Genre
- Budget
- Runtime
- Director
- Main Cast
- Production Company
- Language
- Release Date
- IMDb Rating
- Popularity
- Vote Count
- Revenue (Target)

---

# 🔍 Exploratory Data Analysis

The project includes:

- Revenue Distribution
- Budget vs Revenue
- Genre Analysis
- Release Month Analysis
- Director Performance
- Production Company Comparison
- Correlation Heatmap
- Feature Importance

---

# ⚙️ Machine Learning Pipeline

```
Collect Data
      │
Clean Data
      │
EDA
      │
Feature Engineering
      │
Train Models
      │
Hyperparameter Tuning
      │
Evaluate Models
      │
Explain Predictions
      │
Deploy API
      │
Interactive Dashboard
```

---

# 🤖 Models Used

### Regression

- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### Classification

- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine

---

# 📈 Evaluation Metrics

Regression

- MAE
- RMSE
- R² Score

Classification

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

# 🧠 Explainable AI

This project uses:

- SHAP
- LIME

to explain every prediction and identify the most influential features.

---

# 🌐 Dashboard Features

## Home
- Project Overview
- KPIs

## Prediction
- Revenue Prediction
- Success Probability

## Analytics
- Revenue Trends
- Budget Analysis
- Genre Insights

## Explainability
- SHAP Values
- Feature Importance

## Recommendations
- Best Release Month
- Budget Suggestions
- Similar Movies

---

# 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /predict | POST | Predict Revenue |
| /movies | GET | Movie Information |
| /analytics | GET | Dashboard Data |
| /health | GET | API Status |

---

# 🚀 Installation

```bash
git clone https://github.com/yourusername/CinePredict-AI.git

cd CinePredict-AI

pip install -r requirements.txt
```

Run FastAPI

```bash
uvicorn api.main:app --reload
```

Run Dashboard

```bash
streamlit run dashboard/app.py
```


---

# 👨‍💻 Author

**Aryan Bhoya**

Computer Engineering Student

Passionate about Data Science, Artificial Intelligence, Machine Learning, and Full-Stack Development.

GitHub: https:(https://github.com/aryan01109)
LinkedIn: https:(https://www.linkedin.com/in/bhoya-aryan-1410a3334)

---

## 💙 Thank You

Thank you for visiting **CinePredict AI**! If you have suggestions or feedback, feel free to open an issue or contribute to the project.
