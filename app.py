"""
Smart Spam Classification System
Flask backend application integrating the ML prediction service.
"""

from flask import Flask, jsonify, request
from services.predictor import predict_message

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    """Health check / root endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Smart Spam Classification System",
        "endpoints": {
            "predict": "POST /predict or POST /"
        }
    }), 200


@app.route("/predict", methods=["POST"])
@app.route("/", methods=["POST"])
def predict():
    """
    Prediction endpoint.
    Accepts JSON payload: {"message": "..."} or form data: message="...".
    Returns prediction, spam_probability, and confidence.
    Gracefully handles empty or missing input.
    """
    data = request.get_json(silent=True) or request.form
    message = data.get("message") if data else None

    if message is None or not str(message).strip():
        return jsonify({
            "error": "Input message cannot be empty or blank.",
            "success": False
        }), 400

    try:
        result = predict_message(message)
        return jsonify({
            "success": True,
            "message": result["message"],
            "prediction": result["prediction"],
            "spam_probability": result["spam_probability"],
            "confidence": result["confidence"]
        }), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


def main():
    """Main entry point to run development server."""
    print("Starting Smart Spam Classification System Flask backend...")
    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()
