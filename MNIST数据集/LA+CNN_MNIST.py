import torch
import numpy as np
from torch.optim import Optimizer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random

# 设置随机种子以确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 设置一个固定的随机种子

class LBFGSAdam(Optimizer):
    def __init__(self, params, lr=1e-3, p1=0.9, p2=0.999):
        defaults = dict(lr=lr, p1=p1, p2=p2)
        super(LBFGSAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['r'] = torch.zeros_like(p.data)
                    state['u'] = torch.zeros_like(p.data)

                r, u = state['r'], state['u']
                p1, p2, lr = group['p1'], group['p2'], group['lr']

                state['step'] += 1
                r.mul_(p1).add_(grad, alpha=1 - p1)
                u.mul_(p2).add_(grad, alpha=1 - p2)

                direction = self.lbfgs_direction(r, u)
                p.data.add_(direction, alpha=-lr)

        return loss

    def lbfgs_direction(self, r, u):
        # 实现LBFGS特定的计算
        return r + u

# 定义神经网络
class CNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 需要根据实际输出调整
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)  # 调整展平后的维度
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 设置数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = LBFGSAdam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        def closure():
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')
