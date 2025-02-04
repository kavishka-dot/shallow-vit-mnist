# **Shallow Vision Transformer (ViT) on MNIST**  

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

## **Mathematical Formulation**  

### **Self-Attention**  

\[
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
\]

### **Multi-Head Attention**  

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
\]

### **Transformer Encoder Block**  

\[
X' = \text{MHA}(\text{LayerNorm}(X)) + X
\]

\[
X'' = \text{MLP}(\text{LayerNorm}(X')) + X'
\]

## **Installation & Usage**  

Clone the repository:  
```bash
git clone https://github.com/your-username/shallow-vit-mnist.git
cd shallow-vit-mnist
