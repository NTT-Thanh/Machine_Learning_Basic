"""
Simple Flask API for Customer Churn Prediction.

Run:
    python main.py

API:
    POST /predict
"""

from flask import Flask, request, jsonify

from predict import predict_customer

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Customer Churn Ensemble API",
        "endpoint": "POST /predict",
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "JSON body is required"}), 400

    required = [
        "Age",
        "Balance",
        "Products",
        "Active",
        "Tenure",
        "CreditScore",
        "EstimatedSalary",
    ]

    missing = [feature for feature in required if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing features",
            "missing": missing,
        }), 400

    try:
        result = predict_customer(data)
        return jsonify(result)

    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
