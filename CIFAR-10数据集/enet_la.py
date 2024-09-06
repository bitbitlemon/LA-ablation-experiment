import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.datasets import CIFAR10
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

# Define Enet model
class InitialBlock(nn.Module):
    def __init__(self):
        super(InitialBlock, self).__init__()
        self.main_branch = nn.Conv2d(3, 13, kernel_size=3, stride=2, padding=1, bias=False)
        self.ext_branch = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU(16)

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat((main, ext), 1)
        out = self.bn(out)
        return self.prelu(out)

class RegularBottleneck(nn.Module):
    def __init__(self, in_channels, inter_channels, kernel_size, padding, dropout_prob, asymmetric):
        super(RegularBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU(inter_channels)
        if asymmetric:
            self.conv2 = nn.Sequential(
                nn.Conv2d(inter_channels, inter_channels, (kernel_size, 1), padding=(padding, 0), bias=False),
                nn.Conv2d(inter_channels, inter_channels, (1, kernel_size), padding=(0, padding), bias=False)
            )
        else:
            self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.prelu2 = nn.PReLU(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout2d(dropout_prob)
        self.prelu3 = nn.PReLU(in_channels)
        self.out_prelu = nn.PReLU(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out += identity
        return self.out_prelu(out)

class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super(DownsamplingBottleneck, self).__init__()
        self.main_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.main_bn = nn.BatchNorm2d(out_channels)
        self.main_prelu = nn.PReLU(out_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.prelu1 = nn.PReLU(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.prelu2 = nn.PReLU(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.1)
        self.prelu3 = nn.PReLU(out_channels)
        self.out_prelu = nn.PReLU(out_channels)

    def forward(self, x):
        main = self.main_conv(x)
        main = self.main_bn(main)
        main = self.main_prelu(main)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out += main
        return self.out_prelu(out)

class Enet(nn.Module):
    def __init__(self, num_classes=10):
        super(Enet, self).__init__()
        self.initial_block = InitialBlock()
        self.bottleneck1_0 = DownsamplingBottleneck(16, 16, 64)
        self.bottleneck1_1 = RegularBottleneck(64, 16, 3, 1, 0.1, False)
        self.bottleneck1_2 = RegularBottleneck(64, 16, 3, 1, 0.1, False)
        self.bottleneck1_3 = RegularBottleneck(64, 16, 3, 1, 0.1, False)
        self.bottleneck1_4 = RegularBottleneck(64, 16, 3, 1, 0.1, False)
        self.bottleneck2_0 = DownsamplingBottleneck(64, 32, 128)
        self.bottleneck2_1 = RegularBottleneck(128, 32, 3, 1, 0.1, False)
        self.bottleneck2_2 = RegularBottleneck(128, 32, 3, 1, 0.1, True)
        self.bottleneck2_3 = RegularBottleneck(128, 32, 3, 1, 0.1, False)
        self.bottleneck2_4 = RegularBottleneck(128, 32, 3, 1, 0.1, True)
        self.bottleneck2_5 = RegularBottleneck(128, 32, 3, 1, 0.1, False)
        self.bottleneck2_6 = RegularBottleneck(128, 32, 3, 1, 0.1, True)
        self.bottleneck2_7 = RegularBottleneck(128, 32, 3, 1, 0.1, False)
        self.bottleneck2_8 = RegularBottleneck(128, 32, 3, 1, 0.1, True)
        self.bottleneck2_9 = RegularBottleneck(128, 32, 3, 1, 0.1, False)
        self.bottleneck2_10 = RegularBottleneck(128, 32, 3, 1, 0.1, True)
        self.bottleneck2_11 = RegularBottleneck(128, 32, 3, 1, 0.1, False)
        self.bottleneck3_0 = RegularBottleneck(128, 64, 3, 1, 0.1, False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.initial_block(x)
        x = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)
        x = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)
        x = self.bottleneck2_9(x)
        x = self.bottleneck2_10(x)
        x = self.bottleneck2_11(x)
        x = self.bottleneck3_0(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Data preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = CIFAR10(root='./', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = CIFAR10(root='./', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define Enet model
net = Enet(num_classes=10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = LBFGSAdam(net.parameters(), lr=0.001)

# Define accuracy calculation function
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

# Open file to save metrics
with open("enet-lbfgsadam-cifar.txt", "w") as f:
    # Train the model
    for epoch in range(100):  # Train for multiple epochs
        running_loss = 0.0
        running_accuracy = 0.0
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

            # Calculate accuracy for each batch
            with torch.no_grad():
                outputs = net(inputs)
                accuracy = calculate_accuracy(outputs, labels)
                running_accuracy += accuracy

        # Calculate and save average loss and accuracy for each epoch
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = running_accuracy / len(trainloader)
        f.write(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average Accuracy: {epoch_accuracy:.4f}\n')
        print(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average Accuracy: {epoch_accuracy:.4f}')

    print('Finished Training')
