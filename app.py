
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import html
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="JobShield AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models_rebuilt_test" / "all_three_models_test.joblib"
RESULT_PATH = BASE_DIR / "models_rebuilt_test" / "model_comparison_test.csv"


@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results():
    return pd.read_csv(RESULT_PATH)


models = load_models()
results = load_results()


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
    line-height: 1.8;
    overflow-wrap: anywhere;
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

mark.fake {
    background: #FFD6D6;
    color: #A40000;
    padding: 3px 5px;
    border-radius: 4px;
    font-weight: bold;
}

mark.real {
    background: #C9F7D8;
    color: #126B35;
    padding: 3px 5px;
    border-radius: 4px;
    font-weight: bold;
}

h1, h2, h3 {
    color: #A84478;
}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def get_model_signals(text, model, prediction):
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

        direction = 1 if prediction == 1 else -1
        contributions = values * weights * direction

        indices = np.argsort(contributions)[::-1]

        signals = []

        for index in indices:
            if contributions[index] <= 0:
                continue

            word = feature_names[index]

            if len(word) < 2:
                continue

            if word not in signals:
                signals.append(word)

            if len(signals) >= 8:
                break

        return signals

    except Exception:
        return []


def highlight_evidence(text, signals, prediction):
    safe_text = html.escape(text)

    if not signals:
        return safe_text

    ordered = sorted(signals, key=len, reverse=True)

    patterns = [
        re.escape(word).replace(r"\ ", r"\s+")
        for word in ordered
    ]

    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)",
        re.IGNORECASE
    )

    css_class = "fake" if prediction == 1 else "real"

    return pattern.sub(
        lambda match: (
            f'<mark class="{css_class}">'
            f'{match.group(0)}</mark>'
        ),
        safe_text
    )


def show_prediction(text, model_name):
    model = models[model_name]

    prediction = int(model.predict([text])[0])

    signals = get_model_signals(
        text,
        model,
        prediction
    )

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 FAKE JOB DETECTED")

        st.markdown(
            "### This job posting is classified as potentially fraudulent."
        )

        st.markdown("### 🔴 Suspicious Evidence")

        st.write(
            "The selected model identified the following "
            "text features supporting its Fake Job prediction."
        )

    else:
        st.success("✅ REAL JOB DETECTED")

        st.markdown(
            "### This job posting is classified as likely legitimate."
        )

        st.markdown("### 🟢 Supporting Evidence")

        st.write(
            "The selected model identified the following "
            "text features supporting its Real Job prediction."
        )

    if signals:
        highlighted = highlight_evidence(
            text,
            signals,
            prediction
        )

        st.markdown(
            f"""
            <div class="section-card">
                {highlighted}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Highlighted terms represent model-specific "
            "TF-IDF feature contributions."
        )

        st.markdown("### Important Identified Terms")

        st.write(", ".join(signals))

    else:
        st.info(
            "No clear individual text features could be highlighted."
        )

    st.info(
        "AI predictions are based on learned text patterns. "
        "They do not guarantee the actual legitimacy of a job."
    )


st.sidebar.markdown("# 🌸 JobShield AI")
st.sidebar.caption("AI-Based Fake Job Detection System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home Dashboard",
        "Job Prediction",
        "Model Comparison"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Project")
st.sidebar.caption("Light Pink • White • Cyan")


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
        REAL JOB or FAKE JOB with highlighted evidence.
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="section-card">
        <h3>📊 Model Comparison</h3>
        Compare the performance of three machine
        learning models using percentage-based charts.
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "JobShield AI provides machine learning-based "
        "predictions to support safer job searching."
    )


elif page == "Job Prediction":

    st.markdown(
        '<div class="main-title">🔍 Job Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Analyze job postings using Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("## Enter Job Information")

    job_text = st.text_area(
        "Paste Complete Job Description",
        height=280,
        placeholder=(
            "Enter job title, company details, description, "
            "requirements, salary and other information..."
        ),
        key="job_description"
    )

    st.markdown("### Select Machine Learning Model")

    model_names = list(models.keys())

    default_index = (
        model_names.index("Linear SVM")
        if "Linear SVM" in model_names else 0
    )

    selected_model = st.selectbox(
        "Choose a model",
        model_names,
        index=default_index
    )

    if st.button(
        "✨ Analyze Job Posting",
        use_container_width=True
    ):

        if not job_text.strip():
            st.warning(
                "Please enter a job description first."
            )

        else:
            show_prediction(
                job_text,
                selected_model
            )


elif page == "Model Comparison":

    st.title("📊 Machine Learning Model Comparison")

    st.write(
        "Compare the performance of all three trained "
        "machine learning models."
    )

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    chart_data = results.copy()

    for col in metric_columns:
        if col not in chart_data.columns:
            st.error(f"Missing metric column: {col}")
            st.stop()

        chart_data[col] = (
            pd.to_numeric(chart_data[col]) * 100
        )

    st.markdown("### 📈 Performance Comparison")

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(chart_data))
    width = 0.18

    for i, metric in enumerate(metric_columns):
        bars = ax.bar(
            x + (i - 1.5) * width,
            chart_data[metric],
            width,
            label=metric
        )

        ax.bar_label(
            bars,
            fmt="%.2f%%",
            padding=3,
            fontsize=8
        )

    ax.set_ylabel("Performance (%)")
    ax.set_xlabel("Machine Learning Models")

    ax.set_title("Model Performance Comparison")

    ax.set_xticks(x)
    ax.set_xticklabels(
        chart_data["Model"],
        rotation=10,
        ha="right"
    )

    ax.set_ylim(0, 115)

    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### 📋 Performance Scores")

    display_data = chart_data.copy()

    for col in metric_columns:
        display_data[col] = display_data[col].map(
            lambda value: f"{value:.2f}%"
        )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "All performance values are based on the saved test dataset results."
    )
