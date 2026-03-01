"""
Main engine for model training
"""
from asyncio import log
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

    linear = {
        "model" : model_linear,
        "scores" : scores_linear,
        "mean" : mean_linear,
        "std" : std_linear
    }
    ridge = {
        "model" : model_ridge,
        "scores" : scores_ridge,
        "mean" : mean_ridge,
        "std" : std_ridge,
        "alpha" : best_alpha
    }
    result = {
        "linear" : linear,
        "ridge" : ridge,
        "data" : data_csv,
        "model" : model_path
    }
    return result

def predict(input_data : str, model_path : str = "model.pkl") -> dict:
    result = {
        "model" : model_path,
        "input" : input_data,
        "prediction" : 123.45
    }
    return result