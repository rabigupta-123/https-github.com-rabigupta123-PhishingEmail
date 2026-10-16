import unittest
from pathlib import Path

from phishing_email_analyzer import (
    cross_validation_summary,
    extract_url_features,
    expert_assessment,
    fit_full_model,
    make_classifier,
    phishing_probability,
    predict,
    read_dataset,
    train_model,
)


PROJECT_DIR = Path(__file__).parent
DATASET = PROJECT_DIR / "data" / "emails.csv"


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts, cls.labels = read_dataset(DATASET)
        cls.model, cls.word_vectorizer, cls.char_vectorizer, cls.scaler = fit_full_model(cls.texts, cls.labels)

    def test_dataset_has_two_balanced_classes(self):
        self.assertEqual(len(self.texts), 40)
        self.assertEqual(self.labels.count("Phishing"), self.labels.count("Safe"))

    def test_url_features_capture_suspicious_structure(self):
        features = extract_url_features(["Verify now at http://192.168.1.50/login"])
        self.assertEqual(features.shape, (1, 16))
        self.assertGreater(features[0, 0], 0)
        self.assertGreater(features[0, 4], 0)

    def test_url_features_extended_signals(self):
        features = extract_url_features([
            "Dear customer! Claim your prize now at http://user:pass@claim-win.xyz/login immediately! Enter your banking information."
        ]).toarray()[0]
        # URL count
        self.assertGreater(features[0], 0)
        # @ in url
        self.assertGreater(features[3], 0)
        # high risk domain suffix (.xyz)
        self.assertGreater(features[7], 0)
        # urgent words
        self.assertGreater(features[9], 0)
        # sensitive action words
        self.assertGreater(features[10], 0)
        # dear customer
        self.assertEqual(features[11], 1.0)
        # exclamation count >= 2
        self.assertEqual(features[12], 1.0)
        # banking info
        self.assertEqual(features[13], 1.0)

    def test_model_predicts_both_representative_classes(self):
        messages = [
            "Urgent: verify your password at http://secure-login.example.com immediately.",
            "The team meeting is scheduled for Tuesday at 10 AM in conference room B.",
        ]
        predictions = predict(self.model, self.word_vectorizer, self.char_vectorizer, self.scaler, messages)
        probabilities = phishing_probability(self.model, self.word_vectorizer, self.char_vectorizer, self.scaler, messages)
        self.assertEqual(predictions.tolist(), ["Phishing", "Safe"])
        self.assertGreater(probabilities[0], 0.5)
        self.assertLess(probabilities[1], 0.5)

    def test_phishing_probability_bounds(self):
        probabilities = phishing_probability(
            self.model, self.word_vectorizer, self.char_vectorizer, self.scaler, self.texts
        )
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probabilities))

    def test_cross_validation_is_reproducible_and_high(self):
        summary = cross_validation_summary(self.texts, self.labels)
        self.assertEqual(summary["folds"], 5)
        self.assertGreaterEqual(summary["accuracy_mean"], 0.8)
        self.assertGreaterEqual(summary["f1_macro_mean"], 0.8)

    def test_expert_assessment_explains_high_risk_message(self):
        confidence, risk_level, action, evidence = expert_assessment(
            "Urgent: verify your password at http://192.168.1.50/login", 0.95
        )
        self.assertEqual(confidence, "Very high confidence")
        self.assertEqual(risk_level, "Critical")
        self.assertIn("Do not click", action)
        self.assertGreaterEqual(len(evidence), 3)

    def test_expert_assessment_safe_message(self):
        confidence, risk_level, action, evidence = expert_assessment(
            "Weekly team meeting is tomorrow morning.", 0.05
        )
        self.assertEqual(confidence, "High confidence")
        self.assertEqual(risk_level, "Low")
        self.assertIn("No strong phishing pattern", action)
        self.assertEqual(len(evidence), 0)

    def test_train_model_returns_valid_metrics(self):
        model, w_vec, c_vec, scaler, acc, actual, pred, test_texts = train_model(self.texts, self.labels)
        self.assertGreaterEqual(acc, 0.8)
        self.assertEqual(len(actual), len(pred))
        self.assertEqual(len(test_texts), len(actual))

    def test_make_classifier_structure(self):
        pipeline = make_classifier()
        self.assertIn("classifier", pipeline.named_steps)

    def test_read_dataset_raises_for_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            read_dataset(PROJECT_DIR / "non_existent_file.csv")


if __name__ == "__main__":
    unittest.main()
