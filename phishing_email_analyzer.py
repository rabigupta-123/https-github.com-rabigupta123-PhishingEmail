"""Phishing email detector using Scikit-learn text and URL features."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
IP_URL_PATTERN = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}", re.IGNORECASE)
DOMAIN_PATTERN = re.compile(r"(?:https?://|www\.)([^/\s]+)", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
URGENT_WORDS = {
    "urgent", "verify", "suspend", "password", "invoice", "click", "limited",
    "immediately", "winner", "refund", "security", "confirm", "account",
}
SENSITIVE_ACTION_WORDS = {"login", "sign in", "reset", "verify", "confirm", "pay", "claim", "update"}


def read_dataset(path: Path) -> tuple[list[str], list[str]]:
    """Read a CSV containing `text` and `label` columns."""
    texts, labels = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            text = (row.get("text") or row.get("email") or "").strip()
            label = (row.get("label") or row.get("class") or "").strip().lower()
            if text and label in {"phishing", "safe", "legitimate"}:
                texts.append(text)
                labels.append("Phishing" if label == "phishing" else "Safe")
    if len(set(labels)) < 2:
        raise ValueError("Dataset must contain both phishing and safe labels.")
    return texts, labels


def extract_url_features(texts: Iterable[str]) -> csr_matrix:
    """Extract URL, identity, urgency, and sensitive-action signals."""
    rows = []
    for text in texts:
        lower = text.lower()
        urls = URL_PATTERN.findall(text)
        domains = [match.group(1).lower().rstrip(".,;:!?)]") for match in DOMAIN_PATTERN.finditer(text)]
        url_lengths = [len(url) for url in urls]
        rows.append([
            len(urls),
            max(url_lengths, default=0),
            sum(length > 60 for length in url_lengths),
            sum("@" in url for url in urls),
            sum(bool(IP_URL_PATTERN.search(url)) for url in urls),
            sum(url.count("-") for url in urls),
            len(set(domains)),
            sum(domain.endswith((".top", ".xyz", ".click", ".gq", ".tk")) for domain in domains),
            len(EMAIL_PATTERN.findall(text)),
            sum(lower.count(word) for word in URGENT_WORDS),
            sum(lower.count(word) for word in SENSITIVE_ACTION_WORDS),
            int("dear customer" in lower or "dear user" in lower),
            int(text.count("!") >= 2),
            int(any(token in lower for token in ("credential", "banking information", "card details"))),
            len(text),
            len(text.split()),
        ])
    return csr_matrix(np.asarray(rows, dtype=float))


def train_model(texts: list[str], labels: list[str]) -> tuple[Pipeline, TfidfVectorizer, TfidfVectorizer, StandardScaler, float, np.ndarray, np.ndarray, list[str]]:
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    word_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, max_features=12000, sublinear_tf=True)
    scaler = StandardScaler(with_mean=False)
    train_features = hstack([
        word_vectorizer.fit_transform(train_texts),
        char_vectorizer.fit_transform(train_texts),
        scaler.fit_transform(extract_url_features(train_texts)),
    ]).tocsr()
    test_features = hstack([
        word_vectorizer.transform(test_texts),
        char_vectorizer.transform(test_texts),
        scaler.transform(extract_url_features(test_texts)),
    ]).tocsr()
    model = make_classifier()
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    return model, word_vectorizer, char_vectorizer, scaler, accuracy_score(test_labels, predictions), np.asarray(test_labels), np.asarray(predictions), test_texts


def fit_full_model(texts: list[str], labels: list[str]) -> tuple[Pipeline, TfidfVectorizer, TfidfVectorizer, StandardScaler]:
    """Fit every preprocessing step and the classifier on the supplied training fold."""
    word_vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, max_features=12000, sublinear_tf=True)
    scaler = StandardScaler(with_mean=False)
    features = hstack([
        word_vectorizer.fit_transform(texts),
        char_vectorizer.fit_transform(texts),
        scaler.fit_transform(extract_url_features(texts)),
    ]).tocsr()
    model = make_classifier()
    model.fit(features, labels)
    return model, word_vectorizer, char_vectorizer, scaler


def make_classifier() -> Pipeline:
    """Use calibrated probabilities so the score is a probability estimate, not a raw margin."""
    return Pipeline([
        ("classifier", CalibratedClassifierCV(
            estimator=LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
            method="sigmoid",
            cv=3,
            n_jobs=1,
        )),
    ])


def predict(model: Pipeline, word_vectorizer: TfidfVectorizer, char_vectorizer: TfidfVectorizer, scaler: StandardScaler, texts: list[str]) -> np.ndarray:
    features = hstack([
        word_vectorizer.transform(texts),
        char_vectorizer.transform(texts),
        scaler.transform(extract_url_features(texts)),
    ]).tocsr()
    return model.predict(features)


def phishing_probability(model: Pipeline, word_vectorizer: TfidfVectorizer, char_vectorizer: TfidfVectorizer, scaler: StandardScaler, texts: list[str]) -> np.ndarray:
    """Return the model probability for the Phishing class."""
    features = hstack([
        word_vectorizer.transform(texts),
        char_vectorizer.transform(texts),
        scaler.transform(extract_url_features(texts)),
    ]).tocsr()
    probabilities = model.predict_proba(features)
    phishing_index = list(model.classes_).index("Phishing")
    return probabilities[:, phishing_index]


def expert_assessment(text: str, probability: float) -> tuple[str, str, str, list[dict[str, str]]]:
    """Combine model confidence with interpretable signals for an analyst-facing decision."""
    features = extract_url_features([text]).toarray()[0]
    evidence = []
    if features[0] > 0:
        evidence.append({"Category": "URL analysis", "Evidence": f"{int(features[0])} URL(s) found", "Severity": "High" if features[4] or features[7] else "Medium"})
    if features[4] > 0:
        evidence.append({"Category": "URL analysis", "Evidence": "IP-address URL", "Severity": "Critical"})
    if features[7] > 0:
        evidence.append({"Category": "URL analysis", "Evidence": "High-risk domain suffix", "Severity": "High"})
    if features[9] > 0:
        evidence.append({"Category": "Social engineering", "Evidence": "Urgency or account-verification wording", "Severity": "High"})
    if features[10] > 0:
        evidence.append({"Category": "Requested action", "Evidence": "Login, payment, reset, or claim request", "Severity": "High"})
    if features[13] > 0:
        evidence.append({"Category": "Sensitive data", "Evidence": "Credential or financial-information language", "Severity": "Critical"})
    if probability >= 0.85:
        confidence = "Very high confidence"
    elif probability >= 0.65 or probability <= 0.35:
        confidence = "High confidence"
    else:
        confidence = "Moderate confidence"
    if probability >= 0.5:
        level = "Critical" if any(item["Severity"] == "Critical" for item in evidence) else "High"
        action = "Do not click, reply, or provide information. Report the message and verify the sender through a trusted channel."
    else:
        level = "Low"
        action = "No strong phishing pattern was detected. Still verify unexpected requests through a trusted channel."
    return confidence, level, action, evidence


def cross_validation_summary(texts: list[str], labels: list[str]) -> dict[str, float | int]:
    """Evaluate complete feature fitting inside each stratified fold."""
    folds = min(5, min(labels.count("Phishing"), labels.count("Safe")))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    accuracies, f1_scores = [], []
    for train_indices, validation_indices in splitter.split(texts, labels):
        train_texts = [texts[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        validation_texts = [texts[index] for index in validation_indices]
        validation_labels = [labels[index] for index in validation_indices]
        model, word_vectorizer, char_vectorizer, scaler = fit_full_model(train_texts, train_labels)
        predictions = predict(model, word_vectorizer, char_vectorizer, scaler, validation_texts)
        accuracies.append(accuracy_score(validation_labels, predictions))
        f1_scores.append(classification_report(
            validation_labels, predictions, labels=["Phishing", "Safe"],
            output_dict=True, zero_division=0,
        )["macro avg"]["f1-score"])
    return {
        "folds": folds,
        "accuracy_mean": round(float(np.mean(accuracies)), 4),
        "accuracy_std": round(float(np.std(accuracies)), 4),
        "f1_macro_mean": round(float(np.mean(f1_scores)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify emails as Phishing or Safe.")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data") / "emails.csv")
    parser.add_argument("--text", help="Classify one email instead of only evaluating the dataset.")
    parser.add_argument("--matrix", type=Path, default=Path(__file__).with_name("confusion_matrix.png"))
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("analysis_report.json"))
    args = parser.parse_args()

    texts, labels = read_dataset(args.data)
    model, word_vectorizer, char_vectorizer, scaler, accuracy, actual, predicted, test_texts = train_model(texts, labels)
    print(f"Loaded {len(texts)} emails: {labels.count('Phishing')} phishing, {labels.count('Safe')} safe")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(actual, predicted, labels=["Phishing", "Safe"], zero_division=0))
    matrix = confusion_matrix(actual, predicted, labels=["Phishing", "Safe"])
    cross_validation = cross_validation_summary(texts, labels)
    ConfusionMatrixDisplay(matrix, display_labels=["Phishing", "Safe"]).plot(cmap="Blues", values_format="d")
    plt.title("Phishing Email Detector")
    plt.tight_layout()
    args.matrix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.matrix, dpi=160)
    plt.close()
    print(f"Confusion matrix saved to: {args.matrix}")
    report = {
        "dataset_size": len(texts),
        "class_counts": {"Phishing": labels.count("Phishing"), "Safe": labels.count("Safe")},
        "holdout_size": len(actual),
        "accuracy": round(float(accuracy), 4),
        "roc_auc": round(float(roc_auc_score(actual == "Phishing", phishing_probability(model, word_vectorizer, char_vectorizer, scaler, test_texts))), 4) if len(set(actual)) == 2 else None,
        "confusion_matrix_labels": ["Phishing", "Safe"],
        "confusion_matrix": matrix.tolist(),
        "cross_validation": cross_validation,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Analysis report saved to: {args.report}")
    print(
        f"Cross-validation ({cross_validation['folds']} folds): "
        f"accuracy {cross_validation['accuracy_mean']:.2%} +/- {cross_validation['accuracy_std']:.2%}, "
        f"macro-F1 {cross_validation['f1_macro_mean']:.2%}"
    )

    if args.text:
        prediction = predict(model, word_vectorizer, char_vectorizer, scaler, [args.text])[0]
        probability = phishing_probability(model, word_vectorizer, char_vectorizer, scaler, [args.text])[0]
        print(f"Prediction: {prediction}")
        print(f"Phishing probability: {probability:.2%}")


if __name__ == "__main__":
    main()
