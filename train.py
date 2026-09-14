"""
Train and compare Ensemble Learning models for Customer Churn.

Run:
    python train.py

The script:
1. Loads data/churn.csv
2. Splits train/test data
3. Trains Decision Tree, Bagging, Random Forest,
   AdaBoost, Gradient Boosting, Voting and Stacking
4. Prints evaluation metrics
5. Saves the best model by F1 to ensemble_model.pkl
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "churn.csv"
MODEL_PATH = BASE_DIR / "ensemble_model.pkl"

FEATURES = [
    "Age",
    "Balance",
    "Products",
    "Active",
    "Tenure",
    "CreditScore",
    "EstimatedSalary",
]

TARGET = "Churn"


def build_models():
    """Create all models used in the comparison."""

    decision_tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=12,
        random_state=RANDOM_STATE,
    )

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
        ),
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    adaboost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
        n_estimators=150,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    )

    # Voting combines different models.
    voting = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000)),
            (
                "dt",
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=12,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=7,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ],
        voting="soft",
    )

    # Stacking uses a meta-model to learn how to combine base models.
    stacking = StackingClassifier(
        estimators=[
            (
                "dt",
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=12,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=7,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        stack_method="predict_proba",
        n_jobs=-1,
    )

    return {
        "Decision Tree": decision_tree,
        "Bagging": bagging,
        "Random Forest": random_forest,
        "AdaBoost": adaboost,
        "Gradient Boosting": gradient_boosting,
        "Voting": voting,
        "Stacking": stacking,
    }


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        ),
    }


def main():
    print("=" * 70)
    print("CUSTOMER CHURN - ENSEMBLE LEARNING")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\nDataset: {len(df)} customers")
    print(f"Train:   {len(X_train)}")
    print(f"Test:    {len(X_test)}")
    print(f"Churn rate: {y.mean():.2%}")

    models = build_models()

    results = []
    trained_models = {}

    print("\nTraining models...\n")

    for name, model in models.items():
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        results.append({
            "Model": name,
            **metrics,
        })

        trained_models[name] = model

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("F1", ascending=False)

    print("-" * 70)
    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.3f}".format,
                "Precision": "{:.3f}".format,
                "Recall": "{:.3f}".format,
                "F1": "{:.3f}".format,
                "ROC-AUC": "{:.3f}".format,
            },
        )
    )
    print("-" * 70)

    # Select the best model by F1.
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    package = {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURES,
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(package, file)

    print(f"\nBest model by F1: {best_name}")
    print(f"Saved model: {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
