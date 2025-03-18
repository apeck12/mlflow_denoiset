# Initialize the model
import torch
from model_code.pytorch_model import SimpleNN
from torch import nn

def train() -> nn.Module:
    # Initialize the model. This is just a placeholder for the actual training code, and we'll just use the initial, random weights
    model = SimpleNN()
    return model

def save_model(model: nn.Module, path: str):
    # Save the PyTorch model to a file
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")

def test(path: str):
    model = SimpleNN()
    model.load_state_dict(torch.load(path))

    # Set the model to evaluation mode
    model.eval()

    # Create a valid batch of input tensors with 10 features per input
    input_tensor = torch.randn(3, 5)  # Batch size of 3, 5 features
    print("Input Tensor:", input_tensor)

    # Run a prediction
    output = model(input_tensor)

    print("Prediction:", output)
    
if __name__ == "__main__":
    trained_model = train()
    save_model(trained_model, "model_data/model.pt")
    test("model_data/model.pt")