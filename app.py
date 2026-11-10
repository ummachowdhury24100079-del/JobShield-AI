
import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
from pathlib import Path

# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="JobShield AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models_rebuilt_test" / "all_three_models_test.joblib"
RESULT_PATH = BASE_DIR / "models_rebuilt_test" / "model_comparison_test.csv"

# ==============================
# LOAD MODELS
# ==============================

@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_results():
    return pd.read_csv(RESULT_PATH)

models = load_models()
results = load_results()

# ==============================
# COLOR THEME
# ==============================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #FFF1F7, #F0FCFF);
    color: #334155;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F8C9DD, #C7F3F7);
}

[data-testid="stSidebar"] * {
    color: #334155 !important;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #B84F83;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #64748B;
    font-size: 16px;
    margin-bottom: 25px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #F4C6DA;
    box-shadow: 0 4px 14px rgba(190, 110, 150, 0.10);
    text-align: center;
}

.metric-value {
    font-size: 27px;
    font-weight: bold;
    color: #C05287;
}

.metric-label {
    color: #64748B;
    font-size: 14px;
}

.section-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #C5EDF0;
    margin-bottom: 15px;
}

.stButton > button {
    background: linear-gradient(90deg, #EFA6C8, #83DDE5);
    color: #263746;
    border: none;
    border-radius: 12px;
    font-weight: bold;
    min-height: 45px;
}

.stButton > button:hover {
    border: 1px solid #B84F83;
    color: #263746;
}

h1, h2, h3 {
    color: #A84478;
}

</style>
""", unsafe_allow_html=True)


# ==============================
# HELPER FUNCTIONS
# ==============================

def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def get_signals(text, model, prediction):
    """Find important text features supporting the prediction."""

    try:
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]

        vector = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        values = vector.toarray()[0]

        if hasattr(classifier, "coef_"):
            weights = classifier.coef_[0]

        elif hasattr(classifier, "feature_log_prob_"):
            weights = (
                classifier.feature_log_prob_[1]
                - classifier.feature_log_prob_[0]
            )

        else:
            return []

        contributions = values * weights

        # Class 0 uses the opposite direction
        if prediction == 0:
            contributions = -contributions

        indices = np.argsort(contributions)[::-1]

        signals = []

        for index in indices:
            if contributions[index] <= 0:
                continue

            word = feature_names[index]

            if word not in signals:
                signals.append(word)

            if len(signals) == 5:
                break

        return signals

    except Exception:
        return []


def detect_warnings(text):

    rules = {
        "Upfront Payment": [
            r"registration fee",
            r"processing fee",
            r"pay.*fee",
            r"upfront payment"
        ],
        "Unrealistic Earnings": [
            r"earn.*per day",
            r"guaranteed income",
            r"unlimited earning",
            r"high salary.*no experience"
        ],
        "Sensitive Information": [
            r"bank account",
            r"credit card",
            r"personal password"
        ],
        "Urgent Pressure": [
            r"apply immediately",
            r"limited time",
            r"urgent hiring",
            r"act now"
        ]
    }

    found = []

    for category, patterns in rules.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(category)
                break

    return found


# ==============================
# SIDEBAR NAVIGATION
# ==============================

st.sidebar.markdown("# 🌸 JobShield AI")
st.sidebar.caption("AI-Based Fake Job Detection System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home Dashboard",
        "Job Prediction",
        "Model Comparison",
        "Best Model",
        "Prediction Explanation",
        "Suspicious Indicators"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Project")
st.sidebar.caption("Light Pink • White • Cyan")


# ==============================
# HOME DASHBOARD
# ==============================

if page == "Home Dashboard":

    st.markdown(
        '<div class="main-title">🌸 JobShield AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Based Fake Job Posting Detection System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("## Welcome to JobShield AI 👋")

    st.write(
        "An intelligent machine learning application "
        "designed to identify potentially fraudulent job postings."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Dataset Records", "17,880")

    with c2:
        metric_card("Trained ML Models", "3")

    with c3:
        metric_card("Best Accuracy", "99.27%")

    st.markdown("## ✨ Dashboard Features")

    a, b = st.columns(2)

    with a:
        st.markdown("""
        <div class="section-card">
        <h3>🔍 Job Prediction</h3>
        Analyze job descriptions and classify them as
        REAL JOB or FAKE JOB.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
        <h3>📊 Model Comparison</h3>
        Compare Accuracy, Precision, Recall and F1 Score
        across three machine learning models.
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="section-card">
        <h3>🏆 Best Model</h3>
        Identify the best overall performing model.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
        <h3>🛡️ Fraud Indicators</h3>
        Highlight suspicious phrases and potential warning signs.
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Note: Model predictions are based on learned text patterns. "
        "A prediction is not proof that a job is fraudulent."
    )


# ==============================
# JOB PREDICTION
# ==============================

elif page == "Job Prediction":

    st.title("🔍 Job Posting Prediction")

    st.write(
        "Enter the complete job posting below to analyze it."
    )

    model_names = list(models.keys())

    default_index = (
        model_names.index("Linear SVM")
        if "Linear SVM" in model_names else 0
    )

    selected_model = st.selectbox(
        "Select Machine Learning Model",
        model_names,
        index=default_index
    )

    job_text = st.text_area(
        "Job Description",
        height=260,
        placeholder=(
            "Enter job title, company details, description, "
            "requirements and other job information..."
        )
    )

    if st.button("✨ Analyze Job Posting", use_container_width=True):

        if not job_text.strip():

            st.warning("Please enter a job description first.")

        else:

            model = models[selected_model]
            prediction = int(model.predict([job_text])[0])

            st.session_state["last_text"] = job_text
            st.session_state["last_model"] = selected_model
            st.session_state["last_prediction"] = prediction

            st.markdown("---")

            if prediction == 1:

                st.error("🚨 FAKE JOB DETECTED")

                st.markdown(
                    "### This posting is classified as potentially fraudulent."
                )

            else:

                st.success("✅ REAL JOB CLASSIFICATION")

                st.markdown(
                    "### This posting is classified as likely legitimate."
                )

            st.write("Selected Model:", selected_model)

            st.info(
                "This is an ML-based prediction, not a guarantee "
                "of the employer's legitimacy."
            )


# ==============================
# MODEL COMPARISON
# ==============================

elif page == "Model Comparison":

    st.title("📊 Model Performance Comparison")

    st.write(
        "Performance evaluation of the three trained "
        "machine learning models."
    )

    display_results = results.copy()

    metric_columns = [
        "Accuracy", "Precision", "Recall", "F1 Score"
    ]

    formatted = display_results.copy()

    for col in metric_columns:
        formatted[col] = (
            formatted[col].astype(float) * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        formatted,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("## Performance Chart")

    chart_data = display_results.set_index("Model")[
        metric_columns
    ]

    st.bar_chart(chart_data, y=metric_columns)

    st.caption(
        "All metric values in the table and chart are based "
        "on the saved test results."
    )


# ==============================
# BEST MODEL
# ==============================

elif page == "Best Model":

    st.title("🏆 Best Performing Model")

    best = results.sort_values(
        "F1 Score", ascending=False
    ).iloc[0]

    st.markdown(
        f"""
        <div class="section-card">
        <h2>🌸 {best['Model']}</h2>
        <h1 style="color:#C05287;">
        {best['Accuracy'] * 100:.2f}% Accuracy
        </h1>
        <p>Best overall performance based on F1 Score.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Precision",
            f"{best['Precision'] * 100:.2f}%"
        )

    with c2:
        metric_card(
            "Recall",
            f"{best['Recall'] * 100:.2f}%"
        )

    with c3:
        metric_card(
            "F1 Score",
            f"{best['F1 Score'] * 100:.2f}%"
        )

    st.markdown("## Why Linear SVM?")

    st.write("""
    - It achieved the highest Accuracy among the three models.
    - It achieved the highest F1 Score.
    - It provides a strong balance between Precision and Recall.
    """)

    st.info(
        "Logistic Regression achieved the highest Fake Job Recall. "
        "Therefore, it is also an important model for fraud detection."
    )


# ==============================
# PREDICTION EXPLANATION
# ==============================

elif page == "Prediction Explanation":

    st.title("💡 Prediction Explanation")

    if "last_text" not in st.session_state:

        st.info(
            "First make a prediction from the Job Prediction section."
        )

    else:

        text = st.session_state["last_text"]
        model_name = st.session_state["last_model"]
        prediction = st.session_state["last_prediction"]

        st.write("Model:", model_name)

        st.write(
            "Result:",
            "FAKE JOB" if prediction == 1 else "REAL JOB"
        )

        signals = get_signals(
            text, models[model_name], prediction
        )

        st.markdown("### Important Model Signals")

        if signals:

            st.write(
                "The following words or terms contributed "
                "to the model's prediction:"
            )

            for word in signals:
                st.markdown(f"- **{word}**")

        else:

            st.write(
                "No clear individual feature signals "
                "could be extracted for this prediction."
            )

        st.caption(
            "These are model-based text signals, not proof "
            "of fraud or legitimacy."
        )


# ==============================
# SUSPICIOUS INDICATORS
# ==============================

elif page == "Suspicious Indicators":

    st.title("🛡️ Suspicious Job Indicators")

    st.write(
        "Check the job posting for common warning signs."
    )

    if "last_text" not in st.session_state:

        st.info(
            "First analyze a job posting in the Job Prediction section."
        )

    else:

        text = st.session_state["last_text"]
        warnings = detect_warnings(text)

        if warnings:

            st.warning(
                "Potential warning indicators found:"
            )

            for item in warnings:
                st.markdown(f"- 🚩 **{item}**")

        else:

            st.success(
                "No predefined suspicious phrases were detected."
            )

        st.caption(
            "Absence of these warning signs does not guarantee "
            "that a job posting is safe."
        )

    st.markdown("---")

    st.markdown("### Common Warning Signs")

    st.markdown("""
    - Requests for upfront registration or processing fees.
    - Unrealistic salary promises.
    - Requests for sensitive personal or financial information.
    - Excessive urgency or pressure to apply immediately.
    """)

    st.info(
        "Always verify the employer and job details independently."
    )
