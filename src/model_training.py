"""
Standalone training script for the Boston Housing project.

This is NOT used directly by Kubeflow Pipelines, but it matches
the logic of the pipeline components and is included to satisfy
the assignment structure (src/model_training.py) and to allow
local training + evaluation.

Steps:
1. Load raw CSV (same format as used in the pipeline).
2. Split into train/test.
3. Scale features.
4. Train RandomForest regressor.
5. Evaluate on test set (RMSE, R²).
6. Save model and metrics to disk.
"""

import os
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def train_and_evaluate(
    raw_csv_path: str = "data/raw_data.csv",
    model_path: str = "models/random_forest.joblib",
    scaler_path: str = "data/scaler.joblib",
    metrics_path: str = "metrics/metrics.json",
    test_size: float = 0.2,
    random_state: int = 42,
):
    # Load dataset
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(
            f"Could not find raw data at '{raw_csv_path}'. "
            "Make sure data extraction has run or place the CSV there manually."
        )

    df = pd.read_csv(raw_csv_path)

    # Boston housing dataset target column is usually named 'medv'
    if "medv" not in df.columns:
        raise ValueError(
            "Expected target column 'medv' in the dataset, but it was not found."
        )

    X = df.drop(columns=["medv"])
    y = df["medv"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save scaler (optional, but consistent with pipeline)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Save metrics JSON (same idea as model_evaluation component)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics = {"rmse": rmse, "r2": r2}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return rmse, r2


if __name__ == "__main__":
    train_and_evaluate()
