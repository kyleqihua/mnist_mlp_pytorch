# Import necessary PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set hyperparameters
batch_size = 64  # Number of samples per batch
num_epochs = 10  # Number of training cycles
learning_rate = 0.001  # Step size for optimization

# Define data transformations (preprocessing)
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean and std
    ]
)

# Load MNIST dataset
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define the MLP model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input: 784 (28x28 flattened), Output: 128
        self.fc2 = nn.Linear(128, 64)  # Hidden layer: 128 to 64
        self.fc3 = nn.Linear(64, 10)  # Output layer: 64 to 10 (one per digit)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the 28x28 images into a 784-vector
        x = F.relu(self.fc1(x))  # First layer with ReLU activation
        x = F.relu(self.fc2(x))  # Second layer with ReLU
        x = self.fc3(x)  # Output layer (logits)
        return x


# Initialize model, loss function, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()  # Set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Clear previous gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update weights

        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )

# Evaluation
model.eval()  # Set model to evaluation mode
test_loss = 0
correct = 0
total = 0

with torch.no_grad():  # Disable gradient computation for evaluation
    for data, target in test_loader:
        output = model(data)
        test_loss += criterion(output, target).item() * data.size(0)  # Sum total loss
        total += data.size(0)  # Count total samples
        pred = output.argmax(dim=1, keepdim=True)  # Predicted class
        correct += (
            pred.eq(target.view_as(pred)).sum().item()
        )  # Count correct predictions

average_test_loss = test_loss / total
accuracy = 100.0 * correct / total
print(
    f"\nTest set: Average loss: {average_test_loss:.4f}, "
    f"Accuracy: {correct}/{total} ({accuracy:.0f}%)\n"
)
