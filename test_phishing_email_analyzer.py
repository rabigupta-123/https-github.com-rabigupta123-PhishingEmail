import unittest
from pathlib import Path

from phishing_email_analyzer import (
    cross_validation_summary,
    extract_url_features,
    expert_assessment,
    fit_full_model,
    phishing_probability,
    predict,
    read_dataset,
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


if __name__ == "__main__":
    unittest.main()
