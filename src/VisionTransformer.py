import torch
import torch.nn as nn
from Block import Block

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation.

    This model processes an image by:
    - Splitting it into fixed-size patches
    - Encoding the patches into a lower-dimensional space
    - Adding positional embeddings
    - Passing the embeddings through multiple transformer blocks
    - Applying LayerNorm and a final classification head

    Args:
        img_vec_size (int): The size of the image vector.
        n_embd (int): The embedding dimension.
        block_size (int, optional): The number of transformer blocks. Default is 16.

    Forward Pass:
        imgs -> Patch Embedding
        -> Positional Encoding
        -> Transformer Blocks
        -> Layer Normalization
        -> Classification Head
    """
    def __init__(self, img_vec_size, n_embd, block_size=16):
        super().__init__()
        
        # Linear projection of image patches into embedding space
        self.encoder = nn.Linear(49, 32)  # Input: (patch_size^2), Output: 32-dim embedding
        
        # Positional embedding to retain spatial information
        self.pos_embedding = nn.Linear(49, 32)
        
        # Stack of transformer blocks
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4)
        )
        
        # Final normalization layer
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Classification head (outputs 10 classes)
        self.vit_head = nn.Linear(n_embd, 10)

    def forward(self, imgs):
        """
        Forward pass of the Vision Transformer.

        Args:
            imgs (Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output probability distribution over 10 classes.
        """
        patch_size = 7  # Define patch size
        imgs_patches = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # Convert image to patches
        
        # Reshape patches into a tensor of shape (batch_size, num_patches, patch_size^2)
        imgs_patches = imgs_patches.contiguous().view(64, 16, 49)
        
        # Encode image patches
        x = self.encoder(imgs_patches)
        
        # Add positional embeddings
        x = x + self.pos_embedding(imgs_patches)
        
        # Pass through transformer blocks
        x = self.blocks(x)
        
        # Apply final normalization
        x = self.ln_f(x)
        
        # Pass through the classification head
        x = self.vit_head(x)
        
        # Use the first token as the classification output
        x = x[:, 0]  
        
        # Apply softmax for class probabilities
        x = torch.softmax(x, dim=1)
        
        return x
