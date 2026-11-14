
import streamlit as st
import joblib
import numpy as np
import re
import html
from pathlib import Path

st.set_page_config(
    page_title="JobShield AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models_rebuilt_test" / "all_three_models_test.joblib"

@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)

models = load_models()

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #FFF1F7, #F0FCFF);
    color: #334155;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F8C9DD, #C7F3F7);
}
.main-title {
    text-align: center;
    color: #B84F83;
    font-size: 38px;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #64748B;
    margin-bottom: 25px;
}
.stButton > button {
    background: linear-gradient(90deg, #EFA6C8, #83DDE5);
    color: #263746;
    border: none;
    border-radius: 12px;
    font-weight: bold;
    min-height: 45px;
}
.result-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #F4C6DA;
    margin: 15px 0;
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
</style>
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
        text, model, prediction
    )

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 FAKE JOB DETECTED")
        st.markdown(
            "### This job posting is classified as potentially fraudulent."
        )
        st.markdown("### 🔴 Suspicious Evidence")
        explanation = (
            "The selected model identified these highlighted "
            "terms as signals supporting its Fake Job prediction."
        )

    else:
        st.success("✅ REAL JOB DETECTED")
        st.markdown(
            "### This job posting is classified as likely legitimate."
        )
        st.markdown("### 🟢 Supporting Evidence")
        explanation = (
            "The selected model identified these highlighted "
            "terms as signals supporting its Real Job prediction."
        )

    st.markdown("### Model Evidence")
    st.write(explanation)

    if signals:
        highlighted = highlight_evidence(
            text, signals, prediction
        )

        st.markdown(
            f"""
            <div class="result-card">
                {highlighted}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Highlighted words are model-specific TF-IDF "
            "feature contributions."
        )

    else:
        st.info(
            "No clear individual text features could be highlighted."
        )

    st.info(
        "AI predictions are based on learned text patterns. "
        "They do not guarantee the actual legitimacy of a job."
    )


# SIDEBAR

st.sidebar.markdown("# 🌸 JobShield AI")
st.sidebar.caption("AI-Based Fake Job Detection System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    ["Home Dashboard", "Job Prediction"]
)


# HOME DASHBOARD

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

    c1, c2 = st.columns(2)

    c1.metric("Dataset Records", "17,880")
    c2.metric("Trained ML Models", "3")

    st.markdown("## ✨ Main Feature")

    st.markdown("""
    <div class="result-card">
        <h3>🔍 Job Prediction</h3>
        Analyze job descriptions using three machine learning
        models and understand the text evidence behind predictions.
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "JobShield AI provides machine learning-based "
        "predictions to support safer job searching."
    )


# JOB PREDICTION

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

    job_text = st.text_area(
        "Paste Complete Job Description",
        height=280,
        placeholder=(
            "Enter job title, company details, description, "
            "requirements, salary and other information..."
        )
    )

    st.markdown("### Select Machine Learning Model")

    model_names = list(models.keys())

    selected_model = st.selectbox(
        "Choose a model",
        model_names
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
