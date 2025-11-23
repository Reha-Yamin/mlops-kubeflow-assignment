import os
import json
import subprocess
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from kfp.components import create_component_from_func


def data_extraction(
    repo_url: str,
    dvc_data_path: str = "data/raw_data.csv",
    output_csv_path: str = "data/raw_data_kfp.csv",
) -> str:
    """
    Uses `dvc get` to download the versioned dataset from the Git/DVC repo.

    Inputs:
        repo_url: URL of the Git repo that has DVC + dataset
        dvc_data_path: path to the data file inside that repo
        output_csv_path: where to save the downloaded CSV locally

    Output:
        output_csv_path: path to the downloaded CSV file
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    subprocess.run(
        ["dvc", "get", repo_url, dvc_data_path, "-o", output_csv_path],
        check=True,
    )

    return output_csv_path


def data_preprocessing(
    raw_csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> NamedTuple(
    "PreprocessOutputs",
    [
        ("X_train_path", str),
        ("X_test_path", str),
        ("y_train_path", str),
        ("y_test_path", str),
        ("scaler_path", str),
    ],
):
    """
    Loads the raw CSV, splits into train/test, scales features, and saves to disk.

    Inputs:
        raw_csv_path: CSV file with the Boston housing data
        test_size: fraction of data to use for test set
        random_state: random seed for reproducibility

    Outputs:
        X_train_path, X_test_path, y_train_path, y_test_path, scaler_path
    """
    df = pd.read_csv(raw_csv_path)

    # BostonHousing.csv has target column "medv"
    X = df.drop(columns=["medv"])
    y = df["medv"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("data", exist_ok=True)

    X_train_path = "data/X_train.csv"
    X_test_path = "data/X_test.csv"
    y_train_path = "data/y_train.csv"
    y_test_path = "data/y_test.csv"
    scaler_path = "data/scaler.joblib"

    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(X_train_path, index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    joblib.dump(scaler, scaler_path)

    return X_train_path, X_test_path, y_train_path, y_test_path, scaler_path


def model_training(
    X_train_path: str,
    y_train_path: str,
    n_estimators: int = 100,
    max_depth: int = 5,
    model_path: str = "models/random_forest.joblib",
) -> str:
    """
    Trains a RandomForest regressor and saves the model.

    Inputs:
        X_train_path: path to scaled training features CSV
        y_train_path: path to training targets CSV
        n_estimators: number of trees in the forest
        max_depth: max depth of each tree
        model_path: where to save the trained model

    Output:
        model_path: path to the saved model file
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path, squeeze=True)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    return model_path


def model_evaluation(
    model_path: str,
    X_test_path: str,
    y_test_path: str,
    metrics_path: str = "metrics/metrics.json",
) -> str:
    """
    Evaluates the trained model on the test set and saves metrics.

    Inputs:
        model_path: path to the trained model file
        X_test_path: path to scaled test features CSV
        y_test_path: path to test targets CSV
        metrics_path: JSON file where metrics will be stored

    Output:
        metrics_path: path to the saved metrics JSON file
    """
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path, squeeze=True)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    metrics = {"rmse": rmse, "r2": r2}

    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    return metrics_path


def compile_components():
    """
    Compiles the Python functions above into Kubeflow component YAML files
    under the `components/` directory.
    """
    os.makedirs("components", exist_ok=True)

    create_component_from_func(
        data_extraction,
        base_image="python:3.10",
        output_component_file="components/data_extraction_component.yaml",
    )

    create_component_from_func(
        data_preprocessing,
        base_image="python:3.10",
        output_component_file="components/data_preprocessing_component.yaml",
    )

    create_component_from_func(
        model_training,
        base_image="python:3.10",
        output_component_file="components/model_training_component.yaml",
    )

    create_component_from_func(
        model_evaluation,
        base_image="python:3.10",
        output_component_file="components/model_evaluation_component.yaml",
    )


if __name__ == "__main__":
    compile_components()
