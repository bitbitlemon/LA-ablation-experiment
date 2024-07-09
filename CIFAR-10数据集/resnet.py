import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.models import resnet18

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
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义 ResNet-18 模型
net = resnet18(pretrained=False)
net.fc = nn.Linear(net.fc.in_features, 10)  # 修改最后一层以适应 CIFAR-10 数据集

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 打开文件以保存指标
with open("resnet18-cifar-no-optimizer.txt", "w") as f:
    # 训练模型
    for epoch in range(100):  # 训练多个 epoch
        running_loss = 0.0
        running_iou = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data

            # 零梯度
            net.zero_grad()

            # 前向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 手动更新参数
            with torch.no_grad():
                for param in net.parameters():
                    if param.grad is not None:
                        param -= 0.001 * param.grad  # 这里的0.001是学习率，可以调整

            running_loss += loss.item()

            # 计算每个批次的 IOU
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                intersection = (predicted == labels).sum().item()
                union = len(labels)
                iou = intersection / union
                running_iou += iou

        # 计算并保存每个 epoch 的平均损失和 IOU
        epoch_loss = running_loss / len(trainloader)
        epoch_iou = running_iou / len(trainloader)
        f.write(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average IOU: {epoch_iou:.4f}\n')
        print(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average IOU: {epoch_iou:.4f}')

    print('Finished Training')
