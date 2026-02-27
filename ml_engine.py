"""
Main engine for model training
"""

def train_model(data_csv : str, model_path : str = "model.pkl"):
    result = {
        "data" : data_csv,
        "model" : model_path
    }
    return result