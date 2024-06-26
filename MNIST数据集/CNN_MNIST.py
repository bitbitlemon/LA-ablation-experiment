import torch
import numpy as np
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

# 定义神经网络
class SimpleCNN(nn.Module):
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

# IOU 计算函数
def iou(pred, target, n_classes=10):
    ious = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float().item()
        union = (pred_inds | target_inds).sum().float().item()
        if union == 0:
            ious.append(float('nan'))  # 避免除零错误
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

# 设置数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和学习率
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_iou = 0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播计算梯度
        model.zero_grad()
        loss.backward()

        # 手动更新模型参数
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

        total_loss += loss.item()
        total_iou += iou(outputs, targets)

    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}, IOU: {avg_iou}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')
