import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from sentence_transformers import SentenceTransformer
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Revenue AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class PreReleaseRevenueNN(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # Load preprocessing pipeline
    preprocessor = joblib.load(
        "model/movie_preprocessor.pkl"
    )

    # Dummy data to determine processed feature size
    dummy_data = pd.DataFrame([{

        "budget": 100000000,
        "runtime": 120,
        "release_year": 2027,
        "release_month": 6,
        "genre_count": 2,
        "keyword_count": 5,
        "cast_count": 10,
        "production_company_count": 2,
        "original_language": "en",
        "director": "Unknown"

    }])

    processed_dummy = preprocessor.transform(
        dummy_data
    )

    if hasattr(processed_dummy, "toarray"):

        processed_dummy = (
            processed_dummy.toarray()
        )

    structured_size = processed_dummy.shape[1]

    # Load NLP model
    text_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # MiniLM embedding size
    text_size = 384

    # Final input size
    input_size = (
        structured_size +
        text_size
    )

    # Create neural network
    model = PreReleaseRevenueNN(
        input_size
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(
            "model/movie_revenue_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return (
        preprocessor,
        text_model,
        model
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_comma_list(value):

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def format_money(value):

    if value >= 1_000_000_000:

        return (
            f"${value / 1_000_000_000:.2f}B"
        )

    elif value >= 1_000_000:

        return (
            f"${value / 1_000_000:.2f}M"
        )

    elif value >= 1_000:

        return (
            f"${value / 1_000:.2f}K"
        )

    return f"${value:,.0f}"


def classify_movie(roi):

    if roi < 0:

        return "FLOP"

    elif roi < 1:

        return "AVERAGE"

    elif roi < 3:

        return "HIT"

    else:

        return "BLOCKBUSTER"


def category_description(category):

    if category == "BLOCKBUSTER":

        return "Very strong revenue potential."

    elif category == "HIT":

        return "Positive revenue potential."

    elif category == "AVERAGE":

        return "Moderate revenue potential."

    else:

        return "Revenue may remain below production cost."


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:

    preprocessor, text_model, model = load_models()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        "Unable to load the trained model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎬 Movie Revenue AI")

    st.caption(
        "Pre-release movie revenue prediction"
    )

    st.divider()

    st.subheader("🧭 Navigation")

    st.write("🎬 New Prediction")
    st.write("📊 Prediction History")
    st.write("🧠 Model Information")

    st.divider()

    st.subheader("⚙️ AI Pipeline")

    st.write("📊 Structured Movie Data")
    st.write("📝 Movie Overview")
    st.write("🔤 NLP Embedding")
    st.write("🧠 Deep Neural Network")
    st.write("💰 Revenue Prediction")

    st.divider()

    st.subheader("✨ Model")

    st.info(
        "PyTorch + Sentence Transformer + "
        "Structured Feature Processing"
    )

    st.divider()

    st.caption(
        "⚠️ Predictions are estimates and "
        "not guaranteed box-office results."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🎬 Movie Revenue AI")

st.subheader(
    "Predict movie revenue before release using AI"
)

st.write(
    "Enter the movie information below. "
    "The system combines structured movie data "
    "with NLP analysis of the movie overview."
)

st.divider()


# ============================================================
# TOP INFORMATION CARDS
# ============================================================

info1, info2, info3, info4 = st.columns(4)

with info1:

    st.metric(
        "AI Engine",
        "Deep Learning"
    )

with info2:

    st.metric(
        "NLP Model",
        "MiniLM"
    )

with info3:

    st.metric(
        "Prediction",
        "Revenue"
    )

with info4:

    st.metric(
        "Mode",
        "Pre-Release"
    )


st.divider()


# ============================================================
# MOVIE INFORMATION
# ============================================================

st.header("🎥 Movie Information")

st.caption(
    "Enter information that would normally be available "
    "before the movie is released."
)


# ------------------------------------------------------------
# BASIC INFORMATION
# ------------------------------------------------------------

with st.container(border=True):

    st.subheader("📋 Basic Information")

    col1, col2 = st.columns(2)

    with col1:

        movie_name = st.text_input(
            "Movie Name",
            placeholder="Example: Toxic"
        )

        budget = st.number_input(
            "Production Budget ($)",
            min_value=0.0,
            value=100_000_000.0,
            step=1_000_000.0
        )

        runtime = st.number_input(
            "Runtime (minutes)",
            min_value=1,
            max_value=400,
            value=120
        )

        director = st.text_input(
            "Director",
            placeholder="Example: Christopher Nolan"
        )

    with col2:

        language = st.selectbox(
            "Original Language",
            [
                "en",
                "hi",
                "es",
                "fr",
                "de",
                "ja",
                "ko",
                "zh",
                "other"
            ]
        )

        release_year = st.number_input(
            "Release Year",
            min_value=1900,
            max_value=2100,
            value=2027
        )

        release_month = st.selectbox(
            "Release Month",
            list(range(1, 13)),
            format_func=lambda x:
                pd.Timestamp(
                    2024,
                    x,
                    1
                ).strftime("%B")
        )


# ============================================================
# GENRE
# ============================================================

with st.container(border=True):

    st.subheader("🎭 Genres")

    genres = st.multiselect(
        "Select movie genres",
        [
            "Action",
            "Adventure",
            "Animation",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Family",
            "Fantasy",
            "History",
            "Horror",
            "Music",
            "Mystery",
            "Romance",
            "Science Fiction",
            "Thriller",
            "War",
            "Western"
        ]
    )

    if genres:

        st.write(
            "Selected genres:",
            ", ".join(genres)
        )

    else:

        st.caption(
            "Select at least one genre."
        )


# ============================================================
# CAST / PRODUCTION / KEYWORDS
# ============================================================

with st.container(border=True):

    st.subheader("👥 Cast & Production")

    col1, col2 = st.columns(2)

    with col1:

        cast_input = st.text_input(
            "Main Cast",
            placeholder=(
                "Actor 1, Actor 2, Actor 3"
            )
        )

        keywords_input = st.text_input(
            "Keywords",
            placeholder=(
                "space, future, technology, revenge"
            )
        )

    with col2:

        production_input = st.text_input(
            "Production Companies",
            placeholder=(
                "Company 1, Company 2"
            )
        )

        st.info(
            "Separate multiple values using commas."
        )


# ============================================================
# MOVIE OVERVIEW
# ============================================================

with st.container(border=True):

    st.subheader("📝 Movie Overview")

    overview = st.text_area(
        "Describe the movie story",
        height=180,
        placeholder=(
            "Enter the movie plot, story, "
            "characters, conflict and main idea..."
        )
    )

    if overview:

        word_count = len(
            overview.split()
        )

        st.caption(
            f"Overview length: {word_count} words"
        )


# ============================================================
# FEATURE SUMMARY
# ============================================================

st.subheader("🔍 Feature Summary")

summary_col1, summary_col2, summary_col3, summary_col4 = (
    st.columns(4)
)

current_cast = parse_comma_list(
    cast_input
)

current_keywords = parse_comma_list(
    keywords_input
)

current_production = parse_comma_list(
    production_input
)

current_genres = len(genres)


with summary_col1:

    st.metric(
        "Genres",
        current_genres
    )

with summary_col2:

    st.metric(
        "Cast Members",
        len(current_cast)
    )

with summary_col3:

    st.metric(
        "Keywords",
        len(current_keywords)
    )

with summary_col4:

    st.metric(
        "Production Companies",
        len(current_production)
    )


st.divider()


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.button(
    "🚀 Predict Movie Revenue",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    errors = []

    if not movie_name.strip():

        errors.append(
            "Movie name is required."
        )

    if not director.strip():

        errors.append(
            "Director is required."
        )

    if not overview.strip():

        errors.append(
            "Movie overview is required."
        )

    if budget <= 0:

        errors.append(
            "Production budget must be greater than zero."
        )

    if len(genres) == 0:

        errors.append(
            "Select at least one genre."
        )

    if errors:

        st.error(
            "Please fix the following:"
        )

        for error in errors:

            st.write(
                f"• {error}"
            )

    else:

        with st.spinner(
            "🧠 AI is analyzing your movie..."
        ):

            try:

                # ----------------------------------------
                # FEATURE ENGINEERING
                # ----------------------------------------

                genre_count = len(
                    genres
                )

                keyword_count = len(
                    current_keywords
                )

                cast_count = len(
                    current_cast
                )

                production_company_count = len(
                    current_production
                )

                # ----------------------------------------
                # STRUCTURED DATA
                # ----------------------------------------

                movie_data = pd.DataFrame([{

                    "budget":
                        budget,

                    "runtime":
                        runtime,

                    "release_year":
                        release_year,

                    "release_month":
                        release_month,

                    "genre_count":
                        genre_count,

                    "keyword_count":
                        keyword_count,

                    "cast_count":
                        cast_count,

                    "production_company_count":
                        production_company_count,

                    "original_language":
                        language,

                    "director":
                        director

                }])

                # ----------------------------------------
                # PREPROCESS STRUCTURED DATA
                # ----------------------------------------

                structured_features = (
                    preprocessor.transform(
                        movie_data
                    )
                )

                if hasattr(
                    structured_features,
                    "toarray"
                ):

                    structured_features = (
                        structured_features.toarray()
                    )

                # ----------------------------------------
                # NLP EMBEDDING
                # ----------------------------------------

                text_embedding = (
                    text_model.encode(
                        [overview]
                    )
                )

                # ----------------------------------------
                # FEATURE FUSION
                # ----------------------------------------

                final_features = np.concatenate(
                    [
                        structured_features,
                        text_embedding
                    ],
                    axis=1
                )

                # ----------------------------------------
                # TENSOR
                # ----------------------------------------

                input_tensor = torch.tensor(
                    final_features,
                    dtype=torch.float32
                )

                # ----------------------------------------
                # MODEL PREDICTION
                # ----------------------------------------

                model.eval()

                with torch.no_grad():

                    predicted_log_revenue = (
                        model(
                            input_tensor
                        )
                        .item()
                    )

                # ----------------------------------------
                # CONVERT LOG REVENUE
                # ----------------------------------------

                predicted_revenue = np.expm1(
                    predicted_log_revenue
                )

                # Prevent impossible negative values
                predicted_revenue = max(
                    0,
                    predicted_revenue
                )

                # ----------------------------------------
                # ROI
                # ----------------------------------------

                roi = (
                    predicted_revenue - budget
                ) / budget

                # ----------------------------------------
                # CATEGORY
                # ----------------------------------------

                category = classify_movie(
                    roi
                )

                # ----------------------------------------
                # RESULT OBJECT
                # ----------------------------------------

                result = {

                    "movie":
                        movie_name,

                    "revenue":
                        predicted_revenue,

                    "roi":
                        roi,

                    "category":
                        category,

                    "budget":
                        budget,

                    "runtime":
                        runtime,

                    "genres":
                        genre_count,

                    "cast":
                        cast_count,

                    "keywords":
                        keyword_count,

                    "production":
                        production_company_count,

                    "date":
                        datetime.now().strftime(
                            "%d %b %Y, %I:%M %p"
                        )
                }

                # Save result
                st.session_state.prediction_result = (
                    result
                )

                # Save history
                st.session_state.prediction_history.insert(
                    0,
                    result
                )

                # Keep only latest 10
                st.session_state.prediction_history = (
                    st.session_state
                    .prediction_history[:10]
                )

                st.success(
                    "Prediction completed successfully! 🎉"
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)


# ============================================================
# DISPLAY PREDICTION RESULT
# ============================================================

result = st.session_state.prediction_result


if result is not None:

    st.divider()

    st.header("📊 Prediction Result")

    st.caption(
        f"AI analysis for: **{result['movie']}**"
    )

    # --------------------------------------------------------
    # MAIN RESULT
    # --------------------------------------------------------

    result1, result2, result3 = st.columns(3)

    with result1:

        st.metric(
            "💰 Estimated Revenue",
            format_money(
                result["revenue"]
            )
        )

    with result2:

        st.metric(
            "📈 Estimated ROI",
            f"{result['roi'] * 100:.1f}%"
        )

    with result3:

        st.metric(
            "🎯 Prediction",
            result["category"]
        )


    # --------------------------------------------------------
    # CATEGORY MESSAGE
    # --------------------------------------------------------

    category = result["category"]

    if category == "BLOCKBUSTER":

        st.success(
            "🚀 BLOCKBUSTER — "
            + category_description(category)
        )

    elif category == "HIT":

        st.info(
            "⭐ HIT — "
            + category_description(category)
        )

    elif category == "AVERAGE":

        st.warning(
            "⚠️ AVERAGE — "
            + category_description(category)
        )

    else:

        st.error(
            "📉 FLOP — "
            + category_description(category)
        )


    # --------------------------------------------------------
    # REVENUE VS BUDGET
    # --------------------------------------------------------

    st.subheader("💵 Revenue vs Budget")

    revenue_table = pd.DataFrame({

        "Metric": [
            "Production Budget",
            "Predicted Revenue",
            "Estimated Profit"
        ],

        "Amount": [

            format_money(
                result["budget"]
            ),

            format_money(
                result["revenue"]
            ),

            format_money(
                result["revenue"] -
                result["budget"]
            )

        ]
    })

    st.dataframe(
        revenue_table,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # MOVIE PROFILE
    # --------------------------------------------------------

    st.subheader("🔎 Movie Analysis")

    profile1, profile2 = st.columns(2)

    with profile1:

        st.write("### 🎬 Movie Profile")

        st.write(
            f"**Movie:** {result['movie']}"
        )

        st.write(
            f"**Budget:** "
            f"{format_money(result['budget'])}"
        )

        st.write(
            f"**Runtime:** "
            f"{result['runtime']} minutes"
        )

        st.write(
            f"**Genres:** "
            f"{result['genres']}"
        )

    with profile2:

        st.write("### 🤖 AI Features")

        st.write(
            f"**Cast Members:** "
            f"{result['cast']}"
        )

        st.write(
            f"**Keywords:** "
            f"{result['keywords']}"
        )

        st.write(
            f"**Production Companies:** "
            f"{result['production']}"
        )

        st.write(
            "**NLP:** Sentence Transformer"
        )


    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.subheader("🧠 AI Prediction Pipeline")

    pipeline1, pipeline2, pipeline3, pipeline4 = (
        st.columns(4)
    )

    with pipeline1:

        st.info(
            "1️⃣\n\n"
            "**Movie Data**\n\n"
            "Budget, runtime, "
            "release, genre, cast"
        )

    with pipeline2:

        st.info(
            "2️⃣\n\n"
            "**NLP**\n\n"
            "Movie overview → "
            "384D embedding"
        )

    with pipeline3:

        st.info(
            "3️⃣\n\n"
            "**Feature Fusion**\n\n"
            "Structured + NLP"
        )

    with pipeline4:

        st.info(
            "4️⃣\n\n"
            "**Deep Learning**\n\n"
            "Revenue prediction"
        )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.warning(
        "⚠️ This is an AI-generated estimate. "
        "Actual movie revenue can vary because of "
        "marketing, reviews, competition, distribution, "
        "audience response and other external factors."
    )


# ============================================================
# TABS
# ============================================================

st.divider()

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Prediction History",
        "🧠 Model Information",
        "ℹ️ About Project"
    ]
)


# ============================================================
# HISTORY TAB
# ============================================================

with tab1:

    st.subheader(
        "📊 Recent Predictions"
    )

    if not st.session_state.prediction_history:

        st.info(
            "No predictions yet. "
            "Create your first movie prediction above."
        )

    else:

        history_data = []

        for item in (
            st.session_state.prediction_history
        ):

            history_data.append({

                "Movie":
                    item["movie"],

                "Predicted Revenue":
                    format_money(
                        item["revenue"]
                    ),

                "ROI":
                    f"{item['roi'] * 100:.1f}%",

                "Prediction":
                    item["category"],

                "Budget":
                    format_money(
                        item["budget"]
                    ),

                "Time":
                    item["date"]

            })

        history_df = pd.DataFrame(
            history_data
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button(
            "🗑️ Clear Prediction History"
        ):

            st.session_state.prediction_history = []

            st.rerun()


# ============================================================
# MODEL INFORMATION TAB
# ============================================================

with tab2:

    st.subheader(
        "🧠 Model Architecture"
    )

    st.write(
        "The prediction system combines structured "
        "movie information with natural-language "
        "processing."
    )

    architecture_df = pd.DataFrame({

        "Component": [

            "Structured Features",
            "Preprocessing",
            "NLP Model",
            "Text Embedding",
            "Feature Fusion",
            "Deep Learning",
            "Output"

        ],

        "Technology": [

            "Movie metadata",
            "Scikit-learn",
            "Sentence Transformer",
            "all-MiniLM-L6-v2",
            "NumPy",
            "PyTorch",
            "Revenue Regression"

        ]

    })

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "📐 Neural Network"
    )

    nn_df = pd.DataFrame({

        "Layer": [

            "Input",
            "Dense Layer",
            "Dense Layer",
            "Dense Layer",
            "Output"

        ],

        "Architecture": [

            "Structured + NLP",
            "256 neurons",
            "128 neurons",
            "64 neurons",
            "1 neuron"

        ],

        "Activation": [

            "-",
            "ReLU",
            "ReLU",
            "ReLU",
            "Linear"

        ]

    })

    st.dataframe(
        nn_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "📊 Input Features"
    )

    feature_list = [

        "Production Budget",
        "Runtime",
        "Release Year",
        "Release Month",
        "Genre Count",
        "Keyword Count",
        "Cast Count",
        "Production Company Count",
        "Original Language",
        "Director",
        "Movie Overview"

    ]

    for feature in feature_list:

        st.write(
            f"• {feature}"
        )


# ============================================================
# ABOUT PROJECT TAB
# ============================================================

with tab3:

    st.subheader(
        "🎬 About Movie Revenue AI"
    )

    st.write(
        """
        Movie Revenue AI is a machine learning and
        deep learning project designed to estimate
        the potential revenue of a movie before release.
        """
    )

    st.subheader(
        "🎯 Objective"
    )

    st.write(
        """
        The objective is to use historical movie data
        and movie descriptions to estimate potential
        worldwide revenue.
        """
    )

    st.subheader(
        "🛠️ Technologies"
    )

    technologies = pd.DataFrame({

        "Technology": [

            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "XGBoost",
            "PyTorch",
            "Sentence Transformers",
            "Streamlit"

        ],

        "Purpose": [

            "Programming",
            "Data processing",
            "Numerical computation",
            "Data preprocessing",
            "Machine Learning baseline",
            "Deep Learning",
            "NLP",
            "Web application"

        ]

    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "⚠️ Important"
    )

    st.write(
        """
        The prediction should be treated as an estimate.
        Movie revenue depends on many external factors,
        including marketing, distribution, reviews,
        competition, audience behavior and timing.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎬 Movie Revenue AI • "
    "Machine Learning + Deep Learning + NLP • "
    "Built with Streamlit"
)