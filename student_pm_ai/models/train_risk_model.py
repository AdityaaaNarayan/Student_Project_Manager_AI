"""
train_risk_model.py
--------------------
Trains a small RandomForestClassifier on data/sample_risk_data.csv and
saves it to models/risk_model.pkl for the Risk-Assessment Agent to load.

Run from the project root:
    python data/generate_sample_data.py    # first time only
    python models/train_risk_model.py
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "days_since_last_commit",
    "pct_tasks_overdue",
    "missed_checkins",
    "workload_ratio",
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_risk_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "risk_model.pkl")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_sample_data.py` first, "
            "or replace it with real student-project data using the same columns."
        )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("=== Risk model evaluation (held-out test set) ===")
    print(f"Accuracy : {accuracy_score(y_test, preds):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, preds, target_names=["low/medium", "high risk"]))

    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
