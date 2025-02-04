import torch 
import torch.nn as nn
import torch.nn.functional as F

n_embd = 32
dropout = 0.2

class Head(nn.Module):
    """
    A single head of self-attention.

    This module computes:
    - Key, Query, and Value projections for self-attention
    - Scaled dot-product attention
    - Dropout for regularization

    Args:
        n_embd (int): The embedding dimension of input tokens.
        head_size (int): The size of each attention head.
        dropout (float): Dropout probability to use after softmax.

    Forward Pass:
        x -> Key, Query, Value projections
        -> Compute attention scores
        -> Apply softmax and dropout
        -> Weighted sum of values
    """
    def __init__(self, n_embd, head_size, dropout=0.1):  # Added n_embd and dropout as arguments
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the self-attention head.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, n_embd).

        Returns:
            Tensor: Attention-weighted output of the same shape as input.
        """
        B, T, C = x.shape  # Batch size, sequence length, embedding size
        
        # Compute Key, Query, and Value projections
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # Scale by sqrt(C)
        wei = F.softmax(wei, dim=-1)  # Apply softmax across attention scores
        wei = self.dropout(wei)  # Apply dropout for regularization

        # Compute weighted sum of values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)

        return out
