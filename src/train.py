"""
Trains 4 models for the grocery spoilage project:

1. Router model (Random Forest)      -> predicts category: hardy / normal / fragile
2. Hardy specialist (Logistic Reg.)  -> predicts spoilage risk for hardy items
3. Normal specialist (Random Forest) -> predicts spoilage risk for normal items
4. Fragile specialist (XGBoost)      -> predicts spoilage risk for fragile items

Each model's accuracy / precision / recall / F1 is logged to MLflow so we
can compare runs later.
"""

import pickle

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROUTER_FEATURES = ["temperature", "humidity", "distance_km", "initial_shelf_life"]

SPECIALIST_FEATURES = [
    "temperature",
    "humidity",
    "distance_km",
    "initial_shelf_life",
    "packaging_quality",
]

CATEGORY_NAMES = {0: "hardy", 1: "normal", 2: "fragile"}


def log_metrics(model_name, y_test, y_pred):
    """Calculate basic metrics and log them to MLflow."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    mlflow.log_param("model_name", model_name)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    print(f"{model_name:10s} -> acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
    return acc, prec, rec, f1


def train_router(df):
    """Router learns to guess hardy/normal/fragile from environment + shelf life."""
    X = df[ROUTER_FEATURES]
    y = df["category_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="router_model"):
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        log_metrics("router", y_test, y_pred)

    return model


def train_specialist(df, category, model_type):
    """Train one specialist model on rows belonging to a single category."""
    subset = df[df["category"] == category]
    X = subset[SPECIALIST_FEATURES]
    y = subset["spoilage_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name=f"{category}_model"):
        if model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        elif model_type == "xgboost":
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42,
            )
        else:
            raise ValueError("unknown model_type")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        log_metrics(category, y_test, y_pred)

    return model


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def main():
    mlflow.set_experiment("grocery-spoilage")

    df = pd.read_csv("data/grocery_data.csv")

    print("Training router model...")
    router = train_router(df)
    save_model(router, "models/router.pkl")

    print("\nTraining specialist models...")
    hardy_model = train_specialist(df, "hardy", "logistic_regression")
    save_model(hardy_model, "models/hardy_model.pkl")

    normal_model = train_specialist(df, "normal", "random_forest")
    save_model(normal_model, "models/normal_model.pkl")

    fragile_model = train_specialist(df, "fragile", "xgboost")
    save_model(fragile_model, "models/fragile_model.pkl")

    print("\nAll models saved in models/ folder.")


if __name__ == "__main__":
    main()
