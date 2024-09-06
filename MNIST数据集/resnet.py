import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.models import resnet18

# Set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Data preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Define ResNet-18 model and modify the input layer to fit single-channel images
net = resnet18(pretrained=False)
net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify first conv layer for single-channel input
net.fc = nn.Linear(net.fc.in_features, 10)  # Modify the final layer for MNIST (10 classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Using Adam optimizer

# Open file to save metrics
with open("resnet18-mnist.txt", "w") as f:
    # Train the model
    for epoch in range(100):  # Train for multiple epochs
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get inputs
            inputs, labels = data

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Update parameters
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                accuracy = correct / len(labels)
                running_accuracy += accuracy

        # Calculate and save average loss and accuracy for each epoch
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = running_accuracy / len(trainloader)
        f.write(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average Accuracy: {epoch_accuracy:.4f}\n')
        print(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average Accuracy: {epoch_accuracy:.4f}')

    print('Finished Training')
