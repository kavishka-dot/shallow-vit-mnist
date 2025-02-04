# **Shallow Vision Transformer (ViT) on MNIST**  

![image](https://github.com/user-attachments/assets/1453525e-c3e9-4193-94d3-d4bb69733409)


This project implements a **shallow Vision Transformer (ViT) with 3 encoder blocks** for MNIST classification. Unlike traditional CNNs, ViT **uses self-attention** to model relationships between image patches.  

## **Architecture Overview**  

- **Patch Embedding**:  
  - MNIST images (**28×28**) are split into **16 patches of size 7×7**.  
  - Each patch is **flattened and embedded** using a learnable transformation.  
  - **Positional embeddings** are added to retain spatial structure.  

- **Transformer Encoder Block** (Repeated 3 times):  
  - **Multi-Head Self-Attention (MHA)** captures relationships between patches.  
  - **MLP** processes and refines the feature representations.  
  - **LayerNorm** (applied before MHA and MLP) stabilizes training.  
  - **Residual connections** improve gradient flow and optimization.  

- **Classification**:  
  - Instead of a **[CLS] token**, we use the **first token itself** for classification.  

## **Key Features**  

✅ Uses **self-attention** instead of convolutions.  
✅ **Shallow 3-layer ViT** optimized for simplicity.  
✅ Implements **layer normalization before attention & MLP** (ViT-specific).  
✅ Demonstrates **transformers in image classification** without heavy computation.  


## **Installation & Usage**  

Clone the repository:  
```bash
git clone https://github.com/your-username/shallow-vit-mnist.git
cd shallow-vit-mnist
