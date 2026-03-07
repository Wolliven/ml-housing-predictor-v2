import logging
from numpy import sqrt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
from ml_engine import load_dataset, build_models

X, y = load_dataset("data/california_housing.csv")
model_linear, model_ridge = build_models()

pred_ridge = cross_val_predict(model_ridge, X, y, cv=5)
pred_linear = cross_val_predict(model_linear, X, y, cv=5)

r2_linear = r2_score(y, pred_linear)
r2_ridge = r2_score(y, pred_ridge)
rmse_linear = sqrt(mean_squared_error(y, pred_linear))
rmse_ridge = sqrt(mean_squared_error(y, pred_ridge))
mae_linear = mean_absolute_error(y, pred_linear)
mae_ridge = mean_absolute_error(y, pred_ridge)
print(f"Linear Regression - R²: {r2_linear:.4f}, RMSE: {rmse_linear:.4f}, MAE: {mae_linear:.4f}")
print(f"Ridge Regression - R²: {r2_ridge:.4f}, RMSE: {rmse_ridge:.4f}, MAE: {mae_ridge:.4f}")