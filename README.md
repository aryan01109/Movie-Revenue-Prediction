# 🎬 Movie Revenue AI

An AI-powered **pre-release movie revenue prediction system** that
combines structured machine learning features, Natural Language
Processing (NLP), and Deep Learning to estimate a movie's potential
worldwide revenue before release.

The project is built with **Python, Scikit-learn, XGBoost, PyTorch,
Sentence Transformers, and Streamlit**.

------------------------------------------------------------------------

## 📌 Project Overview

Movie revenue depends on many factors such as production budget, genre,
cast, director, release timing, production companies, and the movie's
story.

This project attempts to estimate potential movie revenue using
information that can be available **before a movie is released**.

### Main idea

``` text
Movie Information
       │
       ├── Structured Features
       │      ├── Budget
       │      ├── Runtime
       │      ├── Release Year
       │      ├── Release Month
       │      ├── Genre
       │      ├── Cast
       │      ├── Director
       │      └── Production Company
       │
       └── Movie Overview
              │
              ▼
        Sentence Transformer
              │
              ▼
        384-D Text Embedding
              │
              ▼
     Feature Fusion / Concatenation
              │
              ▼
      PyTorch Deep Neural Network
              │
              ▼
       Predicted Log Revenue
              │
              ▼
       Estimated Revenue + ROI
              │
              ▼
     Flop / Average / Hit /
          Blockbuster
```

------------------------------------------------------------------------

## 🎯 Objectives

-   Predict movie revenue before release.
-   Use structured movie metadata for prediction.
-   Use NLP to understand the movie overview.
-   Combine structured features with text embeddings.
-   Compare traditional machine learning and deep learning approaches.
-   Provide an easy-to-use Streamlit web application.
-   Estimate ROI from predicted revenue and production budget.
-   Classify movies into revenue categories.

------------------------------------------------------------------------

## ✨ Features

### 🎬 Movie Input

Users can enter:

-   Movie name
-   Production budget
-   Runtime
-   Director
-   Original language
-   Release year
-   Release month
-   Genres
-   Main cast
-   Production companies
-   Keywords
-   Movie overview

### 🤖 AI Prediction

The system provides:

-   Estimated worldwide revenue
-   Estimated ROI
-   Revenue category
-   Revenue vs. budget analysis
-   Prediction history
-   AI pipeline information

### 📊 Prediction Categories

               ROI Category
  ---------------- ----------------
            `< 0%` 🔴 Flop
      `0% – <100%` 🟡 Average
    `100% – <300%` 🔵 Hit
           `≥300%` 🟢 Blockbuster

> These categories are project-defined business rules, not official
> film-industry classifications.

------------------------------------------------------------------------

## 🧠 Machine Learning Architecture

The project uses a multimodal approach.

### 1. Structured Features

The structured model uses:

``` text
budget
runtime
release_year
release_month
genre_count
keyword_count
cast_count
production_company_count
original_language
director
```

### 2. NLP

The movie overview is converted into a numerical representation using:

**Sentence Transformer: `all-MiniLM-L6-v2`**

The model generates a **384-dimensional text embedding**.

### 3. Feature Fusion

The processed structured features and text embedding are concatenated:

``` text
Structured Features
        +
384-D Text Embedding
        ↓
Combined Feature Vector
```

### 4. Deep Neural Network

The final PyTorch model uses:

``` text
Input
  ↓
Dense Layer — 256
  ↓
ReLU
  ↓
Batch Normalization
  ↓
Dropout 30%
  ↓
Dense Layer — 128
  ↓
ReLU
  ↓
Batch Normalization
  ↓
Dropout 20%
  ↓
Dense Layer — 64
  ↓
ReLU
  ↓
Output — 1
```

The model predicts **log-transformed revenue**.

The application converts the prediction back to revenue using:

``` python
np.expm1(predicted_log_revenue)
```

------------------------------------------------------------------------

## 📚 Dataset

The project is based on the **TMDB 5000 Movie Dataset**, using:

``` text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

The two files are combined using the movie identifier.

### Important dataset fields

The movie dataset provides information such as:

-   Budget
-   Revenue
-   Runtime
-   Genres
-   Keywords
-   Popularity
-   Release date
-   Original language
-   Production companies
-   Overview

The credits dataset provides:

-   Cast
-   Crew
-   Director-related information

### Data leakage consideration

For a genuine **pre-release** prediction system, post-release variables
such as final `vote_average`, `vote_count`, and post-release popularity
should not be used as prediction inputs.

The final pre-release model focuses on information that can reasonably
be available before release.

------------------------------------------------------------------------

## 🏗️ Project Structure

``` text
movie-revenue-ai/
│
├── app.py
├── requirements.txt
├── README.md
│
└── model/
    ├── movie_preprocessor.pkl
    └── movie_revenue_model.pth
```

### File descriptions

  File                        Purpose
  --------------------------- -------------------------------------------
  `app.py`                    Streamlit web application
  `requirements.txt`          Python dependencies
  `README.md`                 Project documentation
  `movie_preprocessor.pkl`    Saved Scikit-learn preprocessing pipeline
  `movie_revenue_model.pth`   Saved PyTorch neural network weights

------------------------------------------------------------------------

## 🛠️ Technology Stack

### Programming

-   Python

### Data Science

-   Pandas
-   NumPy

### Machine Learning

-   Scikit-learn
-   XGBoost

### Deep Learning

-   PyTorch

### NLP

-   Sentence Transformers
-   `all-MiniLM-L6-v2`

### Web Application

-   Streamlit

### Model Serialization

-   Joblib
-   PyTorch `.pth`

------------------------------------------------------------------------

## ⚙️ Installation

### 1. Clone the project

``` bash
git clone <your-github-repository-url>
cd movie-revenue-ai
```

### 2. Create a virtual environment

Windows:

``` bash
python -m venv venv
```

Activate it:

``` bash
venv\Scripts\activate
```

Linux/macOS:

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Verify Scikit-learn version

The saved preprocessor was created with:

``` text
scikit-learn 1.9.0
```

Therefore, the local environment should use the same version.

``` bash
python -c "import sklearn; print(sklearn.__version__)"
```

Expected:

``` text
1.9.0
```

------------------------------------------------------------------------

## 🚀 Run the Application

From the project root:

``` bash
streamlit run app.py
```

The application will normally be available at:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

## 🧪 Example Prediction

Example input:

``` text
Movie:
Toxic

Budget:
$100,000,000

Runtime:
120 minutes

Language:
Hindi

Release Year:
2026

Genres:
Action, Thriller

Cast:
Yash, Nayanthara, Kiara Advani

Keywords:
action, crime, gangster, revenge

Director:
Geethu Mohandas
```

The user also provides a movie overview.

The application then produces a result similar to:

``` text
Estimated Revenue: $83.88M
Estimated ROI: -16.1%
Prediction: FLOP
```

> The displayed values are examples. Actual results depend on the
> trained model and input data.

------------------------------------------------------------------------

## 📈 ROI Calculation

ROI is calculated using:

``` text
ROI = (Predicted Revenue - Budget) / Budget
```

For example:

``` text
Budget = $100M
Predicted Revenue = $250M

ROI = ($250M - $100M) / $100M
    = 1.5
    = 150%
```

The application uses the ROI to create the project's revenue category.

------------------------------------------------------------------------

## 🔬 Model Development Pipeline

``` text
TMDB Dataset
     ↓
Data Cleaning
     ↓
Merge Movies + Credits
     ↓
Feature Engineering
     ↓
Remove Invalid / Leakage Features
     ↓
Log Transform Revenue
     ↓
Train/Test Split
     ↓
Structured Feature Preprocessing
     ↓
NLP Text Embeddings
     ↓
Feature Fusion
     ↓
PyTorch Neural Network
     ↓
Model Evaluation
     ↓
Save Model
     ↓
Streamlit Application
```

------------------------------------------------------------------------

## 📊 Model Evaluation

The project evaluates regression models using:

### MAE

Mean Absolute Error:

``` text
MAE = average(|actual - predicted|)
```

Lower is better.

### RMSE

Root Mean Squared Error:

``` text
RMSE = sqrt(mean((actual - predicted)^2))
```

Lower is better.

### R² Score

R² measures how much variance in the target is explained by the model.

``` text
Higher R² → Better
```

The project compares the pre-release XGBoost baseline with the
pre-release multimodal Deep Learning model.

------------------------------------------------------------------------

## ⚠️ Limitations

This project has several limitations.

### Dataset Size

The TMDB 5000 dataset contains roughly 5,000 movies, which is relatively
small for a complex deep learning model.

### Missing Real-World Factors

Revenue can be strongly influenced by:

-   Marketing budget
-   Distribution strategy
-   Number of screens
-   Release competition
-   Reviews
-   Audience sentiment
-   Franchise popularity
-   Regional performance
-   Streaming rights
-   Economic conditions
-   Unexpected events

These factors are not fully represented in the dataset.

### Actor/Director Representation

The current application converts cast and production-company inputs
primarily into counts. It does not yet use a dedicated learned
representation of every actor or production company.

### Prediction Uncertainty

The current model produces a point estimate rather than a calibrated
probability distribution or confidence interval.

Therefore:

``` text
Predicted Revenue ≠ Guaranteed Revenue
```

------------------------------------------------------------------------

## 🔐 Model Files

The trained models are stored separately:

``` text
model/
├── movie_preprocessor.pkl
└── movie_revenue_model.pth
```

Do not retrain the model simply to run the application. The Streamlit
app loads these saved artifacts.

If the preprocessing pipeline is recreated with a different Scikit-learn
version, the `.pkl` file may become incompatible. Keep the training and
application environments aligned.

------------------------------------------------------------------------

## 🖥️ Application Screens

The Streamlit application contains:

### Dashboard

``` text
🎬 Movie Revenue AI
        ↓
Movie Information
        ↓
Prediction
        ↓
Revenue / ROI / Category
```

### Prediction History

Stores recent predictions during the current Streamlit session.

### Model Information

Displays:

-   Structured features
-   NLP model
-   Neural network architecture
-   Technology stack

### About Project

Provides project objectives, methodology, and limitations.

------------------------------------------------------------------------

## 🎓 Academic Value

This project demonstrates practical knowledge of:

-   Data preprocessing
-   Exploratory data analysis
-   Feature engineering
-   Regression
-   XGBoost
-   Deep Learning
-   NLP
-   Transfer learning
-   Model evaluation
-   Model serialization
-   Streamlit application development
-   End-to-end ML deployment concepts

------------------------------------------------------------------------

## 👨‍💻 Author

**Aryan Bhoya**

Computer Engineering Student\
L.D. College of Engineering, Ahmedabad

------------------------------------------------------------------------

## 📜 Disclaimer

This project is developed for **educational, research, and demonstration
purposes**.

Movie revenue predictions are estimates generated by a machine learning
model and should not be treated as guaranteed financial or business
forecasts.

------------------------------------------------------------------------

## ⭐ If You Like This Project

Consider giving the repository a ⭐ on GitHub and sharing it with others
interested in Machine Learning, Deep Learning, NLP, and movie analytics.
