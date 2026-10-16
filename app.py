"""Streamlit review interface for the phishing email detector."""
from pathlib import Path
import sys
from datetime import datetime

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


def message_profile(text: str) -> dict[str, str]:
    """Return analyst-friendly facts that complement the classifier score."""
    features = extract_url_features([text]).toarray()[0]
    return {
        "Characters": f"{len(text):,}",
        "Words": f"{len(text.split()):,}",
        "URLs found": str(int(features[0])),
        "Unique domains": str(int(features[6])),
        "Email addresses": str(int(features[8])),
        "Urgency indicators": str(int(features[9])),
        "Sensitive-action indicators": str(int(features[10])),
    }


def main():
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
        .stApp [data-testid="stCaptionContainer"], .stApp summary,
        .stApp [data-testid="stExpander"] p, .stApp [data-testid="stDataFrame"] * {
            color: #102a43 !important;
        }
        .stApp [data-testid="stAlert"] p, .stApp [data-baseweb="notification"] p { color: #102a43 !important; }
        .stApp textarea { color: #102a43 !important; background: #ffffff !important; }
        .stApp textarea::placeholder { color: #526879 !important; opacity: 1; }
        .stApp [data-testid="stMarkdownContainer"] p { color: #102a43 !important; }
        .stApp [data-testid="stBaseButton-primary"], .stApp [data-testid="stBaseButton-primary"] p { color: #ffffff !important; }
        .block-container { max-width: 1180px; padding-top: 2.5rem; }
        .hero { background: #102a43; color: #f7fbff; padding: 2rem 2.2rem; border-radius: 14px; margin-bottom: 1.4rem; }
        .hero h1 { margin: 0; font-size: 2.4rem; letter-spacing: 0; }
        .hero-subtitle { display: block !important; visibility: visible !important; opacity: 1 !important; color: #d7e6f2 !important; margin: .65rem 0 0; font-size: 1.05rem; line-height: 1.5; }
        .hero h1 { color: #f7fbff !important; }
        .result-safe { border-left: 7px solid #16825d; background: #e8f6ef; padding: 1.1rem 1.3rem; border-radius: 8px; }
        .result-phishing { border-left: 7px solid #c13b3b; background: #fff0ef; padding: 1.1rem 1.3rem; border-radius: 8px; }
        .signal { background: #fff; border: 1px solid #dbe4ea; border-radius: 7px; padding: .65rem .8rem; margin: .35rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []

    st.markdown(
        '<div class="hero"><h1>PhishLens</h1><div class="hero-subtitle">Inspect suspicious email text using language, URL, and social-engineering signals.</div></div>',
        unsafe_allow_html=True,
    )

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
        cv_summary = cross_validation_summary(texts, labels)
        st.metric("Training emails", len(texts))
        st.metric("Phishing / Safe", f"{labels.count('Phishing')} / {labels.count('Safe')}")
        st.metric("Cross-validation accuracy", f"{cv_summary['accuracy_mean']:.1%}")
        st.caption("Probability method: sigmoid-calibrated classifier, 3-fold internal calibration")

    if not scan_clicked:
        st.subheader("Scan result")
        empty_a, empty_b, empty_c = st.columns(3)
        empty_a.metric("Calibrated likelihood", "0.0%")
        empty_b.metric("Confidence level", "Not scanned")
        empty_c.metric("Evidence signals", "0")
        st.caption("No email has been scanned yet. Paste content above and click Start Scan to generate a real analysis.")

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
            scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.scan_history.insert(0, {
                "Time": scan_time,
                "Result": prediction,
                "Likelihood": f"{probability:.1%}",
                "Confidence": confidence,
                "Risk": risk_level,
            })
            if prediction == "Phishing":
                st.error(f"PHISHING DETECTED | {risk_level} RISK")
            else:
                st.success("NO STRONG PHISHING PATTERN DETECTED")
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Calibrated likelihood", f"{probability:.1%}")
            metric_b.metric("Confidence level", confidence)
            metric_c.metric("Evidence signals", len(evidence))
            st.progress(probability, text=f"Model score: {probability:.1%} phishing likelihood")
            st.info(f"Recommended action: {action}")
            confidence_basis = (
                f"The calibrated model estimates a {probability:.1%} phishing likelihood. "
                f"The confidence level is {confidence.lower()} because the score is "
                f"{'well above' if probability >= 0.5 else 'well below'} the decision threshold. "
                f"The model averages {cv_summary['accuracy_mean']:.1%} accuracy across "
                f"{cv_summary['folds']} stratified validation folds."
            )
            st.subheader("Confidence interpretation")
            st.write(confidence_basis)
            st.subheader("Expert evidence review")
            if evidence:
                st.dataframe(evidence, use_container_width=True, hide_index=True)
            else:
                st.success("No high-signal URL, urgency, or sensitive-data indicators were found.")
            with st.expander("All detected indicators"):
                for signal in signals:
                    st.markdown(f"- {signal}")
            with st.expander("Deep message analysis"):
                st.dataframe(
                    [{"Metric": name, "Value": value} for name, value in message_profile(email_text).items()],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "This scan analyzes the supplied text and extracted URL patterns only. "
                    "It does not verify sender identity, domain ownership, redirects, attachments, "
                    "headers, or live reputation data."
                )
                st.subheader("Scan history")
                st.dataframe(st.session_state.scan_history, use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Educational analysis only. Treat the probability as a ranking signal, not a guarantee, and never open a suspicious link to test it.")


if __name__ == "__main__":
    main()
