import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.models import resnet18
from torch.optim import Adam

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# 加载 MNIST 数据集
trainset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# 定义 ResNet-18 模型并调整输入层以适应单通道图像
net = resnet18(pretrained=False)
net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改第一层卷积层以适应单通道输入
net.fc = nn.Linear(net.fc.in_features, 10)  # 修改最后一层以适应 MNIST 数据集

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=0.001)

# 定义 IOU 计算函数
def calculate_iou(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    intersection = (predicted == labels).sum().item()
    union = len(labels)
    iou = intersection / union
    return iou

# 打开文件以保存指标
with open("resnet18-mnist-adam.txt", "w") as f:
    # 训练模型
    for epoch in range(100):  # 训练多个 epoch
        running_loss = 0.0
        running_iou = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data

            # 零梯度
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算每个批次的 IOU
            with torch.no_grad():
                outputs = net(inputs)
                iou = calculate_iou(outputs, labels)
                running_iou += iou

        # 计算并保存每个 epoch 的平均损失和 IOU
        epoch_loss = running_loss / len(trainloader)
        epoch_iou = running_iou / len(trainloader)
        f.write(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average IOU: {epoch_iou:.4f}\n')
        print(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average IOU: {epoch_iou:.4f}')

    print('Finished Training')
