import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import os

from src.VisionTransformer import VisionTransformer   

# Load MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

# Initialize the VisionTransformer model
img_vec_size = 49  # 7x7 patch flattened
n_embd = 32        # Embedding dimension

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VisionTransformer(img_vec_size, n_embd).to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# Initialize list to track accuracy for plotting
train_accuracies = []

losses = []

# Training loop for a few epochs
num_epochs = 20
for epoch in range(num_epochs):
    model.train()  # Set model to training mode

    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()        # Backpropagate the loss

        # Update weights
        optimizer.step()

        # Calculate accuracy for this batch
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print loss every 100 batches
        running_loss += loss.item()
        if i % 100 == 0:
            avg_loss = running_loss / 100
            #print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], Loss: {avg_loss:.4f}")
            running_loss = 0.0
            losses.append(avg_loss)
    # Calculate epoch training accuracy
    epoch_accuracy = 100 * correct / total
    train_accuracies.append(epoch_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Accuracy: {epoch_accuracy:.2f}%")


# Assume model is your trained VisionTransformer instance
model_save_path = os.path.join("models", "vision_transformer.pth")  # Path to save in models/ folder

# Save the model state_dict (recommended way in PyTorch)
torch.save(model.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")

# Plot training accuracy
plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', linestyle='-', color='b')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.show()

# Plot training loss
plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='r')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()