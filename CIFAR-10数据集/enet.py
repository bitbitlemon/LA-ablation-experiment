import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.datasets import CIFAR10

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
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义 Enet 模型
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

net = Enet(num_classes=10)

# 定义损失函数
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001

# 定义 IOU 计算函数
def calculate_iou(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    intersection = (predicted & labels).float().sum()
    union = (predicted | labels).float().sum()
    iou = intersection / union
    return iou.item()

# 打开文件以保存指标
with open("enet-cifar.txt", "w") as f:
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
                    if param.grad is not None:
                        param -= learning_rate * param.grad

            running_loss += loss.item()

            # 计算每个批次的 IOU
            with torch.no_grad():
                iou = calculate_iou(outputs, labels)
                running_iou += iou

        # 计算并保存每个 epoch 的平均损失和 IOU
        epoch_loss = running_loss / len(trainloader)
        epoch_iou = running_iou / len(trainloader)
        f.write(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average IOU: {epoch_iou:.4f}\n')
        print(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average IOU: {epoch_iou:.4f}')

    print('Finished Training')
