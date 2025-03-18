import code
import shutil

import mlflow
import numpy as np
import pandas as pd

from model_code.mlflow_model import MyMLflowModel
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

"""
This script saves an MLflow model with a PyTorch model as a custom Python function.
See https://www.mlflow.org/docs/latest/model/python_model for details.
"""

def main():
    # Using a Pandas DataFrame for the example.  
    # This must match what the `MyMLflowModel.predict()` method expects (in mlflow_model.py)
    input_example = pd.DataFrame(np.random.rand(3,5)) # batch of 3 rows, 5 features

    # The input example, above, will be to infer the MLflow model's input and output signatures (data types),
    # but you can also specify it explicitly, if desired. For full details on specifying MLflow model signatures,
    # see https://mlflow.org/docs/latest/model/signatures/#intro-model-signature-input-example
    # Here, as an example, we define the input and output schema explicitly
    input_schema = Schema([ColSpec("double") for i in range(5)])  # 5 features
    output_schema = Schema([ColSpec("double")])  # Single output column
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Create a dictionary of artifacts
    artifacts = {
        "pytorch_model": "model_data/model.pt",
        # Example of adding an auxiliary file:
        "labels_file": "model_data/labels.txt",
    }

    mlflow_model_path = "mlflow_model"

    shutil.rmtree(mlflow_model_path, ignore_errors=True)
    
    # Save the MLflow model to local filesystem. For method details, see 
    # https://www.mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.save_model
    mlflow.pyfunc.save_model(
        # The path where the MLflow model will be saved
        path=mlflow_model_path,
        # An instance of MLflow model class defined in `mlflow_model.py`
        python_model=MyMLflowModel(),
        # A dictionary that maps logical artifact names to local files, including the PyTorch model weights.
        # These artifact files can be read by the MyMLflowModel() class.
        artifacts=artifacts,
        # The Python dependencies required to run the model
        pip_requirements="requirements.txt",
        # A list of paths to Python directories and/or files that are needed in Python environment to load and run the model
        code_paths=["model_code"],
        # An example input data object that can be used to infer the model's input and output schema. The input example
        # will be stored with the MLflow model so that users of the model can understand the model's expected input format.
        input_example=input_example,
        # An optional MLflow model "signature" to define the model's input and output schema; this is not needed when an
        # input_example is provided
        signature=signature, 
    )

    print(f"MLflow model saved to: {mlflow_model_path}")

if __name__ == "__main__":
    main()
