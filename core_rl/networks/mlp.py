from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) - a simple feed-forward neural network.
    
    Used as the backbone for Q-networks, value functions, and policy networks.
    Consists of linear layers followed by ReLU activation functions.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (128, 128)):
        """
        Create an MLP with specified architecture.
        
        Args:
            input_dim: Number of input features (e.g., 4 for CartPole state)
            output_dim: Number of output values (e.g., 2 for CartPole Q-values)
            hidden_dims: Sizes of hidden layers, default=(128, 128)
        """
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be > 0")

        layers: list[nn.Module] = []
        # Build the network layer by layer:
        # Example: input=4, hidden=(128,128), output=2
        # Creates: Linear(4,128) -> ReLU -> Linear(128,128) -> ReLU -> Linear(128,2)
        dims = (input_dim, *hidden_dims, output_dim)
        
        for idx in range(len(dims) - 1):
            # Add a linear layer connecting current dim to next dim
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            
            # Add ReLU activation after all layers except the final output layer
            if idx < len(dims) - 2:
                layers.append(nn.ReLU())
                
        # nn.Sequential runs layers in order: input -> layer1 -> layer2 -> ... -> output
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Example: (64, 4) for batch of 64 CartPole states
               
        Returns:
            Output tensor of shape (batch_size, output_dim)
            Example: (64, 2) for Q-values of 2 actions
        """
        return self.model(x)
