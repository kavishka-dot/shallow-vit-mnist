import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
from FeedForward import FeedForward

class Block(nn.Module):
    """
    Transformer Block used in the Vision Transformer (ViT).

    This block consists of:
    - Multi-Head Self-Attention (MHSA)
    - Feedforward network (FFN)
    - Layer Normalization applied before both MHSA and FFN
    - Residual connections around both MHSA and FFN

    Args:
        n_embd (int): Embedding dimension size.
        n_head (int): Number of attention heads.

    Forward Pass:
        x -> LayerNorm -> Multi-Head Attention -> Residual Connection
        -> LayerNorm -> Feedforward Network -> Residual Connection
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head  # Size of each attention head

        # Multi-Head Self-Attention
        self.sa = MultiHeadAttention(n_head, head_size)
        
        # Feedforward Network
        self.ffwd = FeedForward(n_embd)
        
        # Layer Normalization for stability
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass through the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, n_embd).

        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        # Apply LayerNorm and Multi-Head Self-Attention, then add residual connection
        x = x + self.sa(self.ln1(x))

        # Apply LayerNorm and Feedforward Network, then add residual connection
        x = x + self.ffwd(self.ln2(x))

        return x
