import torch
import os
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as DataLoader
from src.VisionTransformer import VisionTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

# Define model architecture before loading
model = VisionTransformer(img_vec_size=49, n_embd=32)

# Path to the saved model
model_path = os.path.join("models", "vision_transformer.pth")

# Load the model weights
model.load_state_dict(torch.load(model_path))

# Set model to evaluation mode
model.eval()

correct = 0
total = 0

with torch.no_grad():  # No need to compute gradients for testing
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")