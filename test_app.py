import unittest
from pathlib import Path

from app import load_data, message_profile, risk_signals, train_detector


class AppLogicTests(unittest.TestCase):
    def test_risk_signals_detects_ip_and_sensitive_action(self):
        signals = risk_signals("Urgent: verify your account now at http://192.168.1.1/login")
        self.assertIn("Contains 1 URL(s)", signals)
        self.assertIn("URL uses an IP address", signals)
        self.assertIn("Uses urgency or account-verification language", signals)
        self.assertIn("Requests a sensitive action", signals)

    def test_risk_signals_detects_high_risk_domain_and_credentials(self):
        signals = risk_signals("Claim your prize at http://secure-portal.xyz and enter your card details")
        self.assertIn("URL uses a high-risk domain suffix", signals)
        self.assertIn("Mentions credentials or financial information", signals)

    def test_risk_signals_returns_default_for_safe_message(self):
        signals = risk_signals("Good morning team, please find the quarterly report attached for review.")
        self.assertEqual(signals, ["No strong hand-built risk signal detected"])

    def test_message_profile_metrics(self):
        text = "Contact support@example.com immediately at https://example.com/verify to reset credentials."
        profile = message_profile(text)
        self.assertIn("Characters", profile)
        self.assertIn("Words", profile)
        self.assertEqual(profile["URLs found"], "1")
        self.assertEqual(profile["Email addresses"], "1")
        self.assertGreater(int(profile["Urgency indicators"]), 0)
        self.assertGreater(int(profile["Sensitive-action indicators"]), 0)

    def test_load_data_returns_balanced_dataset(self):
        texts, labels = load_data()
        self.assertEqual(len(texts), 40)
        self.assertEqual(labels.count("Phishing"), 20)
        self.assertEqual(labels.count("Safe"), 20)

    def test_train_detector_returns_fitted_pipeline(self):
        model, word_vec, char_vec, scaler = train_detector()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "predict"))
        self.assertIsNotNone(word_vec)
        self.assertIsNotNone(char_vec)
        self.assertIsNotNone(scaler)


if __name__ == "__main__":
    unittest.main()
