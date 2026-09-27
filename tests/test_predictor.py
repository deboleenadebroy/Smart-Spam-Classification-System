"""
Unit tests for the spam classification inference service.
"""

import os
import sys
import unittest

# Ensure project root is available in module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.predictor import clean_text, predict_message


class TestPredictor(unittest.TestCase):
    """
    Test suite for services.predictor.
    """

    def test_import_and_clean_text(self):
        """Verify text cleaning logic matches notebook specification."""
        raw = "Hello World! Visit http://example.com/now for 100% free prizes..."
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "hello world visit for free prizes")

    def test_ham_prediction(self):
        """Verify a normal ham-like message returns ham with valid probabilities."""
        msg = "Hey, are we still meeting for lunch today at 1pm?"
        result = predict_message(msg)

        self.assertEqual(result["message"], msg)
        self.assertEqual(result["prediction"], "ham")
        self.assertIsInstance(result["spam_probability"], float)
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["spam_probability"], 0.0)
        self.assertLessEqual(result["spam_probability"], 1.0)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_spam_prediction(self):
        """Verify a spam-like message returns spam with valid probabilities."""
        msg = "Congratulations! You have won a free prize. Click now to claim."
        result = predict_message(msg)

        self.assertEqual(result["message"], msg)
        self.assertEqual(result["prediction"], "spam")
        self.assertIsInstance(result["spam_probability"], float)
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["spam_probability"], 0.0)
        self.assertLessEqual(result["spam_probability"], 1.0)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertGreater(result["spam_probability"], 0.5)

    def test_empty_input_raises_value_error(self):
        """Verify empty and blank inputs raise a ValueError."""
        with self.assertRaises(ValueError):
            predict_message("")

        with self.assertRaises(ValueError):
            predict_message("   ")

        with self.assertRaises(ValueError):
            predict_message(None)


if __name__ == "__main__":
    unittest.main()
