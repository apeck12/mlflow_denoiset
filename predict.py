import mlflow
import pandas as pd
import numpy as np

"""
Test prediction using the MLflow package model
"""

BATCH_SIZE = 3
FEATURE_COUNT = 5


def test_local_env(input: pd.DataFrame):
    """
    Test prediction using the MLflow package model running in the local (active) virtual environment.
    """
    
    model = mlflow.pyfunc.load_model("mlflow_model")
    predictions = model.predict(input)
    print(predictions)
    
    
def test_isolated_env(input: pd.DataFrame):
    """
    Test prediction using the MLflow package model in an isolated virtual environment.
    """
    
    predictions = mlflow.models.predict(
        model_uri="mlflow_model",
        input_data=input,
        env_manager="uv",
    )
    print(predictions)
    
    
if __name__ == "__main__":
    input = pd.DataFrame(np.random.randn(BATCH_SIZE, FEATURE_COUNT))
    test_local_env(input)
    test_isolated_env(input)