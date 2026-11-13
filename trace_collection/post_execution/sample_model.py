"""Simple neural network model for post-execution trace collection."""

import torch
import torch.nn as nn


class SampleModel(nn.Module):
    """A simple neural network with linear layer and ReLU activation."""

    def __init__(self, input_size: int = 1024, hidden_size: int = 256 * 1024, output_size: int = 256) -> None:
        """
        Initialize the model.

        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Size of output layer
        """
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor after linear layer and ReLU
        """
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
