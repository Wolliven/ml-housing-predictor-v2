"""
Main engine for model training
"""
from asyncio import log
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
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
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    mean = scores.mean()
    std = scores.std()


    result = {
        "scores" : scores,
        "mean" : mean,
        "std" : std,
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