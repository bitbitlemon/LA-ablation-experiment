import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Optimizer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        dec4 = self.upconv4(bottleneck)
        dec4 = F.pad(dec4, [0, enc4.size(3) - dec4.size(3), 0, enc4.size(2) - dec4.size(2)])
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = F.pad(dec3, [0, enc3.size(3) - dec3.size(3), 0, enc3.size(2) - dec3.size(2)])
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = F.pad(dec2, [0, enc2.size(3) - dec2.size(3), 0, enc2.size(2) - dec2.size(2)])
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = F.pad(dec1, [0, enc1.size(3) - dec1.size(3), 0, enc1.size(2) - dec1.size(2)])
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        return self.final_conv(dec1)

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
        return r + u


# 设置数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
model = UNet(in_channels=1, num_classes=10)  # MNIST数据集是单通道的
criterion = nn.CrossEntropyLoss()
optimizer = LBFGSAdam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(torch.float32)
        targets = targets.to(torch.long)
        
        # 将目标扩展为与输入相同的形状
        targets = targets.view(targets.size(0), 1, 1).expand(targets.size(0), 28, 28)
        
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
torch.save(model.state_dict(), 'unet_model.pth')
