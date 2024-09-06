import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.models import resnet18
from torch.optim.optimizer import Optimizer

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

# Custom LBFGSAdam Optimizer (same as your implementation)
class LBFGSAdam(Optimizer):
    # (same as your implementation)
    pass

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

# Define ResNet-18 model and adjust input layer for single-channel images
net = resnet18(pretrained=False)
net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Modify first conv layer for single-channel input
net.fc = nn.Linear(net.fc.in_features, 10)  # Modify the final layer for MNIST (10 classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = LBFGSAdam(net.parameters(), lr=0.001)

# Define IOU calculation function
def calculate_iou(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    intersection = (predicted == labels).sum().item()
    union = len(labels)
    iou = intersection / union
    return iou

# Open file to save metrics
with open("resnet18-mnist-iou.txt", "w") as f:
    # Train the model
    for epoch in range(100):  # Train for multiple epochs
        running_loss = 0.0
        running_iou = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get input data
            inputs, labels = data

            # Zero gradients
            def closure():
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                return loss

            optimizer.step(closure)
            loss = closure().item()
            running_loss += loss

            # Calculate IOU for each batch
            with torch.no_grad():
                outputs = net(inputs)
                iou = calculate_iou(outputs, labels)
                running_iou += iou

        # Calculate and save average loss and IOU for each epoch
        epoch_loss = running_loss / len(trainloader)
        epoch_iou = running_iou / len(trainloader)
        f.write(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average IOU: {epoch_iou:.4f}\n')
        print(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average IOU: {epoch_iou:.4f}')

    print('Finished Training')
