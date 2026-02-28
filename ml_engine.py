"""
Main engine for model training
"""

def train_model(data_csv : str, model_path : str = "model.pkl"):
    result = {
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