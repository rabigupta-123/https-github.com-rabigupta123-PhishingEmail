# PhishLens | Intelligent Phishing Email Detector

[![Tests](https://github.com/rabigupta-123/https-github.com-rabigupta123-PhishingEmail/actions/workflows/test.yml/badge.svg)](https://github.com/rabigupta-123/https-github.com-rabigupta123-PhishingEmail/actions)
![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?logo=pytest&logoColor=white)

An end-to-end machine learning and behavioral analysis system designed to classify email communications as **Phishing** or **Safe**. PhishLens couples hybrid NLP vectorization and URL structural heuristics with a sigmoid-calibrated classifier and an interactive analyst dashboard.

---

## Key Features

- **Hybrid NLP Pipeline:** Combines word-level TF-IDF n-grams $(1, 2)$ and sublinear character-level n-grams $(3, 5)$ to capture contextual phrases, typosquatting, and obfuscated tokens.
- **Deep URL & Heuristic Profiling:** Analyzes 16 hand-crafted domain, lexical, and structural features including:
  - Absolute URL count and maximum length
  - Embedded IP addresses in URL targets (e.g. `http://192.168.1.1/login`)
  - High-risk top-level domains (`.top`, `.xyz`, `.click`, `.gq`, `.tk`)
  - Embedded user credential tokens (`user:pass@domain`)
  - Social-engineering urgency triggers, sensitive action triggers, and credential harvesting keywords
- **Calibrated Probabilistic Inference:** Employs class-balanced Logistic Regression wrapped with `CalibratedClassifierCV` (sigmoid method) to transform raw margins into genuine, well-calibrated posterior probabilities.
- **Robust Model Validation:** Evaluated using holdout splits and 5-fold stratified cross-validation ($97.5\% \pm 5.0\%$ accuracy, macro-F1 $> 0.97$), exporting confusion matrices and JSON telemetry reports.
- **Analyst-Grade Web UI:** Features **PhishLens**, a high-contrast Streamlit review dashboard featuring real-time risk classification, confidence tiers, evidence tables, deep message metrics, and local session scan history.
- **Automated CI/CD:** Complete GitHub Actions pipeline executing the 17-test suite across Python 3.10, 3.11, and 3.12 on every push and pull request.

---

## Architecture Overview

```
Email Text / Raw Message
          │
          ├──► Word TF-IDF Vectorizer (ngram 1-2) ──────┐
          │                                             │
          ├──► Char TF-IDF Vectorizer (ngram 3-5) ──────┼──► Feature Matrix (CSR)
          │                                             │           │
          └──► URL & Heuristic Signals ──► StandardScaler┘           │
                                                                    ▼
                                                       CalibratedClassifierCV
                                                      (LogisticRegression + Sigmoid)
                                                                    │
                                                ┌───────────────────┴───────────────────┐
                                                ▼                                       ▼
                                       Phishing Probability                     Expert Evidence Engine
                                         (0.0% - 100.0%)                       (Indicators & Tiers)
                                                │                                       │
                                                └───────────────────┬───────────────────┘
                                                                    ▼
                                                        PhishLens Analyst Dashboard
```

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── test.yml                 # GitHub Actions CI workflow (Python 3.10, 3.11, 3.12)
├── data/
│   └── emails.csv                   # Labeled dataset for training and cross-validation
├── analysis_report.json             # Automated evaluation metrics and holdout performance
├── app.py                           # PhishLens Streamlit interactive web application
├── confusion_matrix.png             # Visual model performance artifact
├── phishing_email_analyzer.py       # Core classifier, feature extraction, and CLI tool
├── requirements.txt                 # Project dependencies
├── test_app.py                      # Unit test suite for web application logic & signals
└── test_phishing_email_analyzer.py  # Unit test suite for model, vectorizers, and heuristics
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/rabigupta-123/https-github.com-rabigupta123-PhishingEmail.git
cd https-github.com-rabigupta123-PhishingEmail
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Run the Command-Line Analyzer

Evaluate the dataset and generate the classification report, confusion matrix, and JSON metrics:

```bash
python phishing_email_analyzer.py
```

Classify a specific email string directly via CLI:

```bash
python phishing_email_analyzer.py --text "URGENT: Verify your bank account immediately at http://192.168.1.50/login"
```

### Launch the PhishLens Web Dashboard

```bash
streamlit run app.py
```

Once started, open your browser at `http://localhost:8501`. Paste the subject or body of any suspicious email into the analyzer and click **Start Scan** to view:
- **Calibrated Phishing Likelihood** percentage and confidence tier
- **Recommended Action** (e.g., immediate quarantine or verification)
- **Expert Evidence Table** outlining detected IP URLs, suspicious TLDs, and urgency cues
- **Deep Message Telemetry** (word count, URL breakdown, domain count)
- **Scan History** for local auditing during your session

---

## Testing

The repository contains an automated 17-test suite covering dataset validation, feature extraction, probability bounds, model training, cross-validation, and app logic.

### Run All Tests with `unittest`

```bash
python -m unittest discover -v
```

### Run All Tests with `pytest`

```bash
pytest -v
```

### Run Test Suites Individually

```bash
# Core detector tests (11 tests)
python test_phishing_email_analyzer.py

# Web application logic tests (6 tests)
python test_app.py
```

---

## CI/CD Pipeline

Continuous integration is handled via GitHub Actions in [.github/workflows/test.yml](.github/workflows/test.yml). Every push and pull request triggers matrix builds that:
1. Initialize Python 3.10, 3.11, and 3.12 environments
2. Install pinned dependencies from `requirements.txt`
3. Execute all unit tests via `python -m unittest discover -v`
4. Verify tests via `pytest -v`
5. Execute end-to-end training and report generation (`phishing_email_analyzer.py`)

---

## Security Notice & Ethics

- **Safe Local Analysis:** PhishLens performs purely static lexical, structural, and NLP analysis of message text and URL strings. It **never** connects to or dereferences any links present in scanned content.
- **Demonstration Scope:** The included dataset is designed for educational and demonstration purposes. In production security operations (SOC), model probabilities should serve as an input signal alongside SPF/DKIM/DMARC authentication, sender reputation telemetry, and sandboxed attachment analysis.
