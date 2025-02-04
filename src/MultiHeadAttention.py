import torch
import torch.nn as nn
from Head import Head

n_embd = 32
dropout = 0.2

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention module.

    This module performs:
    - Multiple self-attention operations in parallel (each using a `Head`)
    - Concatenation of outputs from all heads
    - A final linear projection followed by dropout

    Args:
        n_embd (int): The embedding dimension of input tokens.
        num_heads (int): The number of attention heads.
        dropout (float): Dropout probability to use after the final projection.

    Forward Pass:
        x -> Multiple attention heads (parallel computation)
        -> Concatenate outputs
        -> Apply final linear projection
        -> Apply dropout
    """
    def __init__(self, n_embd, num_heads, dropout=0.1):  # Added n_embd and dropout as arguments
        super().__init__()
        head_size = n_embd // num_heads  # Ensure each head has equal size

        # Create multiple self-attention heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(num_heads)])

        # Final projection layer
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the multi-head attention module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, n_embd).

        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        # Compute self-attention for each head and concatenate results
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Shape: (B, T, n_embd)
        
        # Apply final linear transformation and dropout
        out = self.dropout(self.proj(out))  # Shape: (B, T, n_embd)

        return out

