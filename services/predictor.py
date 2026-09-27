"""
Smart Spam Classification System - Inference Service
Provides reusable prediction functions using the trained model artifacts.
"""

import os
import re
import joblib

# Determine base directory (project root is parent of services/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "spam_classifier.joblib")
VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")

_model = None
_vectorizer = None
_spam_class_index = None


def _load_artifacts():
    """
    Load saved model and vectorizer artifacts if not already in memory.
    """
    global _model, _vectorizer, _spam_class_index
    if _model is None or _vectorizer is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Model artifacts not found in {ARTIFACTS_DIR}. "
                "Ensure spam_classifier.joblib and tfidf_vectorizer.joblib exist."
            )
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _spam_class_index = list(_model.classes_).index("spam")


def clean_text(text: str) -> str:
    """
    Clean raw text message to match the notebook preprocessing stage.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_message(message: str) -> dict:
    """
    Predict whether an input message is spam or ham.

    Parameters:
        message (str): The raw text message to classify.

    Returns:
        dict: A dictionary containing:
            - message (str): The original input message.
            - prediction (str): 'spam' or 'ham'.
            - spam_probability (float): Probability between 0 and 1.
            - confidence (float): Highest class confidence score between 0 and 1.

    Raises:
        ValueError: If input message is empty or whitespace.
    """
    if message is None or not str(message).strip():
        raise ValueError("Input message cannot be empty or blank.")

    raw_message = str(message).strip()
    cleaned = clean_text(raw_message)

    _load_artifacts()

    transformed = _vectorizer.transform([cleaned])
    prediction = _model.predict(transformed)[0]
    probabilities = _model.predict_proba(transformed)[0]

    spam_prob = float(probabilities[_spam_class_index])
    confidence = float(max(probabilities))

    return {
        "message": raw_message,
        "prediction": str(prediction),
        "spam_probability": spam_prob,
        "confidence": confidence,
    }
