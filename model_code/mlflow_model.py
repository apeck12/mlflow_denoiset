from typing import List
import mlflow
from pandas import DataFrame
import torch

from model_code.pytorch_model import SimpleNN

"""
This file defines a custom MLflow model that wraps a PyTorch model.
See https://www.mlflow.org/docs/latest/model/python_model for details.
"""

class MyMLflowModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """
        This method is called when the model is loaded. It loads the PyTorch model.
        """
        # Load the PyTorch model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNN()
        self.model.load_state_dict(torch.load(context.artifacts["pytorch_model"], map_location=self.device))
        self.model.eval()  # Set the model to evaluation mode

        # Example of loading an auxiliary file (if needed)
        with open(context.artifacts["labels_file"], "r") as f:
            self.labels = [line.strip() for line in f]

    def _preprocess(self, model_input: DataFrame) -> torch.FloatTensor:
        """
        Preprocesses input data before feeding it to the PyTorch model.
        """
        # model_input is expected to be a Pandas DataFrame, so access the data using the column name
        input_data = model_input.to_numpy()

        # Convert to a PyTorch tensor.  Adjust dtype and device as needed.
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        return input_tensor

    def _postprocess(self, prediction: torch.FloatTensor) -> List[str]:
        """
        Postprocesses the model's output.
        """
        # Example: Convert probabilities to class labels
        prediction_array = prediction.cpu().numpy()
        result = [
            self.labels[0] if 0 <= value < 0.5 else self.labels[1]
            for value in prediction_array
        ]
        return result

    def predict(self, context, model_input: DataFrame) -> List[str]:
        """
        Makes predictions using the loaded PyTorch model.
        """
        with torch.no_grad():  # Disable gradient calculation
            input_tensor = self._preprocess(model_input)
            prediction = self.model(input_tensor)
            output = self._postprocess(prediction)
            return output
