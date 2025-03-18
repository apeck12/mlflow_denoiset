import torch.nn as nn

# Define a simple 2-layer neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 1),  # Input layer with 5 features, output layer with 1 unit
            nn.Sigmoid()       # Sigmoid activation for binary classification
        )


    def forward(self, x):
        return self.model(x)

