import code
import shutil

import mlflow
import numpy as np

from model_code.mlflow_model import DenoiseMLFLowModel
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

"""
This script saves an MLflow model with a PyTorch model as a custom Python function.
See https://www.mlflow.org/docs/latest/model/python_model for details.
"""

def main():
    # Using a Pandas DataFrame for the example.  
    # This must match what the `MyMLflowModel.predict()` method expects (in mlflow_model.py)
    input_example = np.random.rand(300,300,200).astype(np.float32)

    # The input example, above, will be to infer the MLflow model's input and output signatures (data types),
    # but you can also specify it explicitly, if desired. For full details on specifying MLflow model signatures,
    # see https://mlflow.org/docs/latest/model/signatures/#intro-model-signature-input-example
    # Here, as an example, we define the input and output schema explicitly
    input_schema = Schema([TensorSpec(np.dtype(np.float32), shape=[-1,-1,-1])])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), shape=[-1,-1,-1])])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Create a dictionary of artifacts
    artifacts = {
        "pytorch_model": "model_data/phantom.pth",
    }

    mlflow_model_path = "mlflow_model"

    shutil.rmtree(mlflow_model_path, ignore_errors=True)
    
    mlflow.pyfunc.save_model(
        path=mlflow_model_path,
        python_model=DenoiseMLFLowModel(),
        artifacts=artifacts,
        pip_requirements="requirements.txt",
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
