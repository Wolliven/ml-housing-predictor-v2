"""
Main engine for model training
"""
import json
import logging as log
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import pickle as pkl


def train_model(data_csv : str, model_path : str = "model.pkl"):
    if not model_path:
        model_path = "model.pkl"
    if not data_csv.endswith(".csv"):
        raise ValueError("Invalid data file format. Please provide a CSV file.")
    if not model_path.endswith(".pkl"):
        raise ValueError("Invalid model file format. Please provide a .pkl file.")
    
    df = pd.read_csv(data_csv)
    missing_rows = df.isna().any(axis=1).sum()
    if missing_rows > 0:
        log(f"Warning: {missing_rows} row/s with missing values will be dropped.")
    df = df.dropna()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    model_linear = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    #Hyperparameter tuning for Ridge Regression using RidgeCV
    model_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=[0.001,0.1, 1.0, 10.0, 100.0], cv=5))
    ])
    model_ridge.fit(X, y)
    best_alpha = model_ridge.named_steps["model"].alpha_

    model_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=best_alpha))
    ])

    scores_ridge = cross_val_score(model_ridge, X, y, cv=5, scoring="r2")
    scores_linear = cross_val_score(model_linear, X, y, cv=5, scoring="r2")
    mean_linear = scores_linear.mean()
    std_linear = scores_linear.std()
    mean_ridge = scores_ridge.mean()
    std_ridge = scores_ridge.std()

    if abs(mean_ridge - mean_linear) < (std_linear + std_ridge) / 2:
        selection = "tie"
        model = model_linear
    elif mean_ridge > mean_linear:
        selection = "ridge"
        model = model_ridge
    else:
        selection = "linear"
        model = model_linear
    
    model.fit(X, y)

    linear = {
        "mean" : mean_linear,
        "std" : std_linear
    }
    ridge = {
        "mean" : mean_ridge,
        "std" : std_ridge,
    }
    result = {
        "selection" : selection,
        "model" : model,
        "linear" : linear,
        "ridge" : ridge,
        "data" : data_csv,
        "model_path" : model_path,
        "features" : X.columns.tolist()
    }

    with open(model_path, "wb") as f:
        pkl.dump(result, f)

    return result

def predict(input_data : str, model_path : str = None, output_path : str = None) -> None:
    if not model_path:
        model_path = "model.pkl"
    if not (input_data.endswith(".json") or input_data.endswith(".csv")):
        raise ValueError("Invalid prediction input file format. Please provide a JSON or CSV file.")
    if not model_path.endswith(".pkl"):
        raise ValueError("Invalid model file format. Please provide a .pkl file.")
    try:
        with open(model_path, "rb") as f:
            model_data = pkl.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if input_data.endswith(".csv"):
        df = pd.read_csv(input_data)
    else:
        with open(input_data, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Invalid JSON format. Please provide a JSON object or an array of JSON objects.")

    missing_rows = df.isna().any(axis=1).sum()
    if missing_rows > 0:
        print(f"Warning: {missing_rows} row/s with missing values will be dropped.")
    df = df.dropna()
    expected = model_data["features"]
    missing = [feat for feat in expected if feat not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in the input data: {', '.join(missing)}")
    input_data = df[expected]
    prediction = model_data["model"].predict(input_data)

    model_type = "ridge" if model_data["selection"] == "ridge" else "linear"

    if not output_path:
        output_path = "predictions.json"
    if not (output_path.endswith(".json") or output_path.endswith(".csv")):
        raise ValueError("Invalid output file format. Please provide a JSON or CSV file.")
    output_df = df.copy()
    output_df["PredictedMedHouseVal"] = prediction
    if output_path.endswith(".csv"):
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_json(output_path, orient="records", indent=2)
        
    return output_path