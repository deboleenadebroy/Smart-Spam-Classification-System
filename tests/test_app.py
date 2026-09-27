"""
Unit tests for the Flask application backend.
"""

import unittest
from app import app


class TestFlaskBackend(unittest.TestCase):
    """
    Test suite for Flask endpoints.
    """

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_get_root(self):
        """Verify GET / returns 200 and healthy status."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")

    def test_valid_prediction_ham(self):
        """Verify valid ham message returns 200 and correct fields."""
        response = self.client.post("/predict", json={
            "message": "Hey, let's catch up over coffee this afternoon."
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("prediction"), "ham")
        self.assertIn("spam_probability", data)
        self.assertIn("confidence", data)
        self.assertIn("message", data)
        self.assertGreaterEqual(data["spam_probability"], 0.0)
        self.assertLessEqual(data["spam_probability"], 1.0)

    def test_valid_prediction_spam(self):
        """Verify valid spam message returns 200 and predicts spam."""
        response = self.client.post("/predict", json={
            "message": "Congratulations! You have won a free prize. Click now to claim."
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("prediction"), "spam")
        self.assertGreater(data["spam_probability"], 0.5)

    def test_empty_input(self):
        """Verify empty input returns 400 with an error message."""
        response = self.client.post("/predict", json={"message": ""})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

        response_blank = self.client.post("/predict", json={"message": "   "})
        self.assertEqual(response_blank.status_code, 400)

        response_none = self.client.post("/predict", json={})
        self.assertEqual(response_none.status_code, 400)

    def test_expected_prediction_fields(self):
        """Verify all expected prediction fields are present in response."""
        response = self.client.post("/predict", json={
            "message": "URGENT! Call 09066362220 to claim your award."
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        required_fields = ["message", "prediction", "spam_probability", "confidence", "success"]
        for field in required_fields:
            self.assertIn(field, data)


if __name__ == "__main__":
    unittest.main()
