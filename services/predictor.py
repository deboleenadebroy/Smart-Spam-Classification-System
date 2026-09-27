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


# Suspicious indicator patterns for lightweight hybrid detection
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+|(?:\b[a-zA-Z0-9-]+\.(?:com|org|net|xyz|top|info|site|online|link|ru|cn|co|io)\b(?:\/\S*)?)",
    re.IGNORECASE,
)

URGENT_SECURITY_PATTERN = re.compile(
    r"\b(?:urgent(?:ly)?|immediate(?:ly)?|action required|act now|alert|warning|"
    r"flagged|suspended|locked|blocked|restricted|frozen|deactivated|"
    r"unauthorized|compromised|fraud(?:ulent)?|security notice|security alert|unusual activity)\b",
    re.IGNORECASE,
)

ACCOUNT_VERIFY_PATTERN = re.compile(
    r"\b(?:verify(?:ing|ed|ication)?|validate|validation|confirm(?:ation)?|"
    r"log(?:in|ged in)?|sign(?:in|ed in)?|auth(?:enticate)?|"
    r"update (?:your )?(?:account|details|info|password|security)|"
    r"reset (?:your )?(?:password|pin)|bank(?:ing)? account|credit card)\b",
    re.IGNORECASE,
)

PRIZE_REWARD_PATTERN = re.compile(
    r"\b(?:congratulations?|winner|won\b|free prize|cash prize|bonus caller|lottery|"
    r"claim (?:now|your|free|reward)|prize guaranteed|selected to receive)\b",
    re.IGNORECASE,
)


def analyze_risk_signals(text: str) -> tuple[list[str], bool, str]:
    """
    Detect suspicious indicators and evaluate if a strong combination exists.

    Returns:
        tuple: (risk_signals: list[str], is_strong_suspicious: bool, reason: str)
    """
    signals = []
    has_url = bool(URL_PATTERN.search(text))
    has_urgent = bool(URGENT_SECURITY_PATTERN.search(text))
    has_verify = bool(ACCOUNT_VERIFY_PATTERN.search(text))
    has_prize = bool(PRIZE_REWARD_PATTERN.search(text))

    if has_url:
        signals.append("url_detected")
    if has_urgent:
        signals.append("urgent_security_language")
    if has_verify:
        signals.append("account_verification_request")
    if has_prize:
        signals.append("prize_reward_lure")

    is_strong = False
    reason = ""

    # Combination 1: URL + Urgent/Security Alert + Account Verification Request
    if has_url and has_urgent and has_verify:
        is_strong = True
        reason = "High-risk phishing alert: URL detected with urgent security alert and account verification request."
    # Combination 2: URL + Account Verification / Login request
    elif has_url and has_verify:
        is_strong = True
        reason = "High-risk credential harvesting: URL detected requesting account verification or login."
    # Combination 3: URL + Prize / Reward lure
    elif has_url and has_prize:
        is_strong = True
        reason = "High-risk spam lure: URL detected with prize or reward claim."
    # Combination 4: Urgent Security Alert + Bank Account Verification (smishing / fraud without URL)
    elif has_urgent and has_verify and ("bank" in text.lower() or "account" in text.lower()):
        is_strong = True
        reason = "High-risk security alert: Urgent warning combined with bank account verification request."

    return signals, is_strong, reason


# Default decision threshold for the machine learning pipeline
DEFAULT_SPAM_THRESHOLD = 0.50


def predict_message(message: str, threshold: float = DEFAULT_SPAM_THRESHOLD) -> dict:
    """
    Predict whether an input message is spam or ham using a hybrid pipeline:
    1. Evaluates raw text for strong suspicious/phishing indicators (URLs, account alerts, etc.).
    2. Runs cleaned text through the trained TF-IDF + Logistic Regression model.
    3. Combines ML probability with explainable risk signals for the final verdict.

    Parameters:
        message (str): The raw text message to classify.
        threshold (float): Probability cutoff for the ML classifier (default: 0.50).

    Returns:
        dict: A dictionary containing:
            - message (str): The original input message.
            - prediction (str): 'spam' or 'ham'.
            - spam_probability (float): Model posterior probability between 0 and 1.
            - confidence (float): Classification confidence score between 0 and 1.
            - risk_signals (list): List of detected risk signal tags.
            - reason (str): Human-readable explanation of the verdict.

    Raises:
        ValueError: If input message is empty or whitespace.
    """
    if message is None or not str(message).strip():
        raise ValueError("Input message cannot be empty or blank.")

    raw_message = str(message).strip()
    cleaned = clean_text(raw_message)

    _load_artifacts()

    transformed = _vectorizer.transform([cleaned])
    probabilities = _model.predict_proba(transformed)[0]

    spam_prob = float(probabilities[_spam_class_index])

    # Extract risk signals and evaluate combinations
    risk_signals, is_strong_suspicious, reason = analyze_risk_signals(raw_message)

    if is_strong_suspicious:
        prediction = "spam"
        confidence = max(0.85, 1.0 - (1.0 - spam_prob) * 0.5)
        if not reason:
            reason = "High-risk suspicious signals detected."
    elif spam_prob >= threshold:
        prediction = "spam"
        confidence = spam_prob
        reason = "Classified as spam by machine learning model."
    else:
        prediction = "ham"
        confidence = float(1.0 - spam_prob)
        reason = "Classified as legitimate by machine learning model."

    return {
        "message": raw_message,
        "prediction": prediction,
        "spam_probability": spam_prob,
        "confidence": confidence,
        "risk_signals": risk_signals,
        "reason": reason,
    }

