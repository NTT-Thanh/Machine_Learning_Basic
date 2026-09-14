"""
Predict churn using the trained ensemble model.

Run:
    python predict.py
"""

import pickle
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ensemble_model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def predict_customer(customer):
    package = load_model()

    model = package["model"]
    features = package["features"]

    X = pd.DataFrame([customer])[features]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]

    return {
        "model": package["model_name"],
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probability), 4),
    }


if __name__ == "__main__":

    customer = {
        "Age": 45,
        "Balance": 150000,
        "Products": 1,
        "Active": 0,
        "Tenure": 3,
        "CreditScore": 600,
        "EstimatedSalary": 80000,
    }

    result = predict_customer(customer)

    print("=" * 50)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 50)
    print(f"Model:             {result['model']}")
    print(f"Prediction:        {result['prediction']}")
    print(f"Churn probability: {result['churn_probability']:.2%}")
