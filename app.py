"""Streamlit review interface for the phishing email detector."""
from pathlib import Path
import sys

import streamlit as st

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from phishing_email_analyzer import (  # noqa: E402
    cross_validation_summary,
    extract_url_features,
    expert_assessment,
    fit_full_model,
    phishing_probability,
    predict,
    read_dataset,
)

st.set_page_config(
    page_title="PhishLens | Email Analyzer",
    page_icon="P",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f4f7f9; color: #102a43; }
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label,
    .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stCaptionContainer"] { color: #102a43 !important; }
    .stApp [data-testid="stAlert"] p, .stApp [data-baseweb="notification"] p { color: #102a43 !important; }
    .stApp [data-testid="stDataFrame"] { color: #102a43 !important; }
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    .hero { background: #102a43; color: #f7fbff; padding: 2rem 2.2rem; border-radius: 14px; margin-bottom: 1.4rem; }
    .hero h1 { margin: 0; font-size: 2.4rem; letter-spacing: 0; }
    .hero p { color: #c9d8e6 !important; margin: .55rem 0 0; font-size: 1.05rem; }
    .result-safe { border-left: 7px solid #16825d; background: #e8f6ef; padding: 1.1rem 1.3rem; border-radius: 8px; }
    .result-phishing { border-left: 7px solid #c13b3b; background: #fff0ef; padding: 1.1rem 1.3rem; border-radius: 8px; }
        .signal { background: #fff; border: 1px solid #dbe4ea; border-radius: 7px; padding: .65rem .8rem; margin: .35rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>PhishLens</h1><p>Inspect suspicious email text using language, URL, and social-engineering signals.</p></div>',
    unsafe_allow_html=True,
)

@st.cache_data

def load_data():
    return read_dataset(PROJECT_DIR / "data" / "emails.csv")

@st.cache_resource

def train_detector():
    texts, labels = load_data()
    return fit_full_model(texts, labels)


def risk_signals(text: str) -> list[str]:
    lower = text.lower()
    features = extract_url_features([text]).toarray()[0]
    signals = []
    if features[0] > 0:
        signals.append(f"Contains {int(features[0])} URL(s)")
    if features[4] > 0:
        signals.append("URL uses an IP address")
    if features[7] > 0:
        signals.append("URL uses a high-risk domain suffix")
    if features[9] > 0:
        signals.append("Uses urgency or account-verification language")
    if features[10] > 0:
        signals.append("Requests a sensitive action")
    if features[13] > 0 or any(word in lower for word in ("credential", "banking", "card details")):
        signals.append("Mentions credentials or financial information")
    return signals or ["No strong hand-built risk signal detected"]


left, right = st.columns([1.55, 1], gap="large")
with left:
    st.subheader("Email to scan")
    email_text = st.text_area(
        "Paste the email content below",
        height=260,
        placeholder="Paste the subject and body here...",
        label_visibility="collapsed",
    )
    scan_clicked = st.button("Start Scan", type="primary", use_container_width=True)
    st.caption("The scan analyzes text and URLs locally. It does not open links.")

with right:
    st.subheader("Model status")
    texts, labels = load_data()
    st.metric("Training emails", len(texts))
    st.metric("Phishing / Safe", f"{labels.count('Phishing')} / {labels.count('Safe')}")
    st.metric("Cross-validation accuracy", f"{cross_validation_summary(texts, labels)['accuracy_mean']:.1%}")

if scan_clicked:
    if not email_text.strip():
        st.warning("Paste an email before starting the scan.")
    else:
        with st.spinner("Analyzing email text and URL signals..."):
            model, word_vectorizer, char_vectorizer, scaler = train_detector()
            prediction = predict(model, word_vectorizer, char_vectorizer, scaler, [email_text])[0]
            probability = float(phishing_probability(model, word_vectorizer, char_vectorizer, scaler, [email_text])[0])
            signals = risk_signals(email_text)
        confidence, risk_level, action, evidence = expert_assessment(email_text, probability)
        if prediction == "Phishing":
            st.error(f"PHISHING DETECTED | {risk_level} RISK")
        else:
            st.success("NO STRONG PHISHING PATTERN DETECTED")
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Phishing probability", f"{probability:.1%}")
        metric_b.metric("Assessment confidence", confidence)
        metric_c.metric("Evidence signals", len(evidence))
        st.progress(probability, text=f"Model score: {probability:.1%} phishing likelihood")
        st.info(f"Recommended action: {action}")
        st.subheader("Expert evidence review")
        if evidence:
            st.dataframe(evidence, use_container_width=True, hide_index=True)
        else:
            st.success("No high-signal URL, urgency, or sensitive-data indicators were found.")
        with st.expander("All detected indicators"):
            for signal in signals:
                st.markdown(f"- {signal}")

st.divider()
st.caption("Educational analysis only. Treat the probability as a ranking signal, not a guarantee, and never open a suspicious link to test it.")
