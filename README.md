# Phishing Email Detection Model

A Scikit-learn project that classifies email text as **Phishing** or **Safe**.

## Features

- Word and character TF-IDF features for email text, including misspellings and obfuscated wording
- URL count, URL length, IP-address URL, `@` URL, domain diversity, high-risk TLD, sender-address, urgency, sensitive-action, credential, punctuation, and length features
- Logistic Regression classifier with class balancing
- Stratified holdout accuracy, per-class precision/recall/F1, ROC-AUC, and confusion matrix
- Five-fold stratified cross-validation with accuracy variation and macro-F1
- JSON analysis output saved as `analysis_report.json`
- Single-email prediction with phishing probability

## Run

```powershell
cd "$HOME\Downloads\phishing_email_analyzer_task3"
py -m pip install -r requirements.txt
py phishing_email_analyzer.py
py phishing_email_analyzer.py --text "Urgent: verify your account at http://login.example.com"
py -m unittest discover -v
```

The CSV must contain `label` and `text` columns. Labels can be `Phishing`, `Safe`, or `Legitimate`.

The included 40-row CSV is a demonstration dataset, not evidence of real-world accuracy. For a credible result, replace it with a substantially larger dataset containing varied phishing and legitimate emails, keep a sender/domain-aware test split, and inspect false negatives. Do not open links from an email just to test the model.

This is an educational detector, not a replacement for email security controls. Model probabilities are ranking signals, not guarantees.
