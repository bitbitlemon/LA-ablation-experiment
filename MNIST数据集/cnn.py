import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

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

classes = tuple(str(i) for i in range(10))

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Input channel changed from 3 to 1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # MNIST input size is 28x28, size after two pooling layers and convolutions is 4x4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = SimpleCNN()

# Define the loss function
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001

# Define accuracy calculation function
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

# Open a file to save metrics
with open("cnn-mnist.txt", "w") as f:
    # Train the model
    for epoch in range(100):  # Train for multiple epochs
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get input data
            inputs, labels = data

            # Zero gradients
            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Manually update parameters
            with torch.no_grad():
                for param in net.parameters():
                    param -= learning_rate * param.grad

            running_loss += loss.item()

            # Calculate accuracy for each batch
            with torch.no_grad():
                accuracy = calculate_accuracy(outputs, labels)
                running_accuracy += accuracy
        
        # Calculate and save average loss and accuracy for each epoch
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = running_accuracy / len(trainloader)
        f.write(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average Accuracy: {epoch_accuracy:.4f}\n')
        print(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average Accuracy: {epoch_accuracy:.4f}')

    print('Finished Training')
