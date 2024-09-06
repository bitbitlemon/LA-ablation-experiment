import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.optimizer import Optimizer
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

# Custom LBFGSAdam optimizer
class LBFGSAdam(Optimizer):
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, history_size=10, max_grad_norm=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, history_size=history_size, max_grad_norm=max_grad_norm)
        super(LBFGSAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            history_size = group['history_size']
            max_grad_norm = group['max_grad_norm']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['old_dirs'] = []
                    state['old_stps'] = []

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = betas

                state['step'] += 1

                if state['step'] > 1:
                    y = grad - state['prev_grad']
                    s = p.data - state['prev_p_data']
                    y_flat, s_flat = y.view(-1), s.view(-1)
                    if y_flat.dot(s_flat) > 1e-10:
                        if len(state['old_dirs']) >= history_size:
                            state['old_dirs'].pop(0)
                            state['old_stps'].pop(0)
                        state['old_dirs'].append(y_flat)
                        state['old_stps'].append(s_flat)

                q = grad.view(-1)
                alphas = []
                for i in range(len(state['old_dirs']) - 1, -1, -1):
                    s, y = state['old_stps'][i], state['old_dirs'][i]
                    alpha = s.dot(q) / (y.dot(s) + eps)
                    q -= alpha * y
                    alphas.append(alpha)

                r = q
                if len(state['old_dirs']) > 0:
                    s, y = state['old_stps'][-1], state['old_dirs'][-1]
                    r *= y.dot(s) / (y.dot(y) + eps)

                for i in range(len(state['old_dirs'])):
                    s, y = state['old_stps'][i], state['old_dirs'][i]
                    beta = y.dot(r) / (s.dot(y) + eps)
                    r += s * (alphas.pop() - beta)

                r = r.view_as(grad)
                exp_avg.mul_(beta1).add_(r, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(r, r, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr / denom

                torch.nn.utils.clip_grad_norm_([p], max_grad_norm)

                with torch.no_grad():
                    p.data.add_(-step_size * exp_avg)

                state['prev_grad'] = grad.clone()
                state['prev_p_data'] = p.data.clone()

        return loss

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

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Input channel changed to 1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Adjusted fully connected layer input dimensions
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

# Open a file to save metrics
with open("cnn-la-mnist.txt", "w") as f:
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
