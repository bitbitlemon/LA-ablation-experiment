import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

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

classes = tuple(str(i) for i in range(10))

# 定义简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 输入通道数从3改为1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # MNIST的输入尺寸是28x28，经过两次池化和卷积后尺寸为4x4
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

# 定义损失函数
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001

# 定义IOU计算函数
def calculate_iou(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    intersection = (predicted & labels).float().sum()
    union = (predicted | labels).float().sum()
    iou = intersection / union
    return iou.item()

# 打开文件以保存指标
with open("cnn-mnist.txt", "w") as f:
    # 训练模型
    for epoch in range(100):  # 训练多个 epoch
        running_loss = 0.0
        running_iou = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data

            # 零梯度
            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 手动更新参数
            with torch.no_grad():
                for param in net.parameters():
                    param -= learning_rate * param.grad

            running_loss += loss.item()

            # 计算每个批次的IOU
            with torch.no_grad():
                iou = calculate_iou(outputs, labels)
                running_iou += iou
        
        # 计算并保存每个 epoch 的平均损失和 IOU
        epoch_loss = running_loss / len(trainloader)
        epoch_iou = running_iou / len(trainloader)
        f.write(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average IOU: {epoch_iou:.4f}\n')
        print(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average IOU: {epoch_iou:.4f}')

    print('Finished Training')
