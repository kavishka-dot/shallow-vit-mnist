import torch
import torch.nn as nn
import torch.nn.functional as F

dropout = 0.2

class FeedForward(nn.Module):
    """
    A simple two-layer feedforward network with ReLU activation.

    This module applies:
    - A linear transformation expanding the embedding size (`4 * n_embd`)
    - A ReLU non-linearity
    - A second linear transformation projecting back to `n_embd`
    - A dropout layer to prevent overfitting

    Args:
        n_embd (int): Size of the input and output embeddings.
        dropout (float): Dropout probability to use after the second linear layer.

    Forward Pass:
        x -> Linear (expand) -> ReLU -> Linear (project) -> Dropout
    """
    def __init__(self, n_embd, dropout=0.1):  # Added dropout as an argument with a default value
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand dimension
            nn.ReLU(),  # Non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Project back to original size
            nn.Dropout(dropout)  # Dropout for regularization
        )

    def forward(self, x):
        """
        Forward pass through the feedforward network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, n_embd).

        Returns:
            Tensor: Transformed tensor of the same shape as input.
        """
        return self.net(x)
