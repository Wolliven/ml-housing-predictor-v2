from numpy import sqrt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
import pickle as pkl
from ml_engine import load_dataset, build_models, add_features

#Baseline model evaluation using cross-validation predictions
X, y = load_dataset("data/california_housing.csv")
model_linear, model_ridge = build_models()

def analyze_models(X : pd.DataFrame, y : pd.Series, model_linear : Pipeline, model_ridge : GridSearchCV) -> None:
    pred_ridge = cross_val_predict(model_ridge, X, y, cv=5)
    pred_linear = cross_val_predict(model_linear, X, y, cv=5)

    r2_linear = r2_score(y, pred_linear)
    r2_ridge = r2_score(y, pred_ridge)
    rmse_linear = sqrt(mean_squared_error(y, pred_linear))
    rmse_ridge = sqrt(mean_squared_error(y, pred_ridge))
    mae_linear = mean_absolute_error(y, pred_linear)
    mae_ridge = mean_absolute_error(y, pred_ridge)

    errors_linear = pred_linear - y
    errors_ridge = pred_ridge - y
    abs_errors_linear = abs(errors_linear)
    abs_errors_ridge = abs(errors_ridge)

    worst_linear_idx = abs_errors_linear.sort_values(ascending=False).head(10).index
    worst_ridge_idx = abs_errors_ridge.sort_values(ascending=False).head(10).index

    print(f"Linear Regression - R²: {r2_linear:.4f}, RMSE: {rmse_linear:.4f}, MAE: {mae_linear:.4f}")
    print(f"Ridge Regression - R²: {r2_ridge:.4f}, RMSE: {rmse_ridge:.4f}, MAE: {mae_ridge:.4f}")

    print("\nLinear error stats:")
    print("mean:", errors_linear.mean())
    print("std:", errors_linear.std())
    print("min:", errors_linear.min())
    print("max:", errors_linear.max())

    print("\nRidge error stats:")
    print("mean:", errors_ridge.mean())
    print("std:", errors_ridge.std())
    print("min:", errors_ridge.min())
    print("max:", errors_ridge.max())

    print("\nWorst Linear predictions:")
    print(X.loc[worst_linear_idx])
    print(y.loc[worst_linear_idx])

    print("\nWorst Linear predictions:")
    print(X.loc[worst_linear_idx])
    print(y.loc[worst_linear_idx])

#Baseline model evaluation
#analyze_models(X, y, model_linear, model_ridge)

#Features
X = add_features(X)

#analyze_models(X, y, model_linear, model_ridge)

def analyze_coefficients(model_path : str) -> None:
    model_path = "model.pkl"

    with open(model_path, "rb") as f:
        model_data = pkl.load(f)
        model = model_data["model"]
    coefficients = model.named_steps["model"].coef_
    feature_names = model_data["features"]
    feature_coefficients = pd.Series(coefficients, index=feature_names)
    print("\nFeature coefficients:")
    print(feature_coefficients.sort_values(key=lambda x: abs(x), ascending=False))