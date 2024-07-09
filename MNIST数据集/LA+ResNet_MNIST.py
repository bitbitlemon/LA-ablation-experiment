import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.models import resnet18
from torch.optim.optimizer import Optimizer

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

# 自定义 LBFGSAdam 优化器
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
optimizer = LBFGSAdam(net.parameters(), lr=0.001)

# 定义 IOU 计算函数
def calculate_iou(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    intersection = (predicted == labels).sum().item()
    union = len(labels)
    iou = intersection / union
    return iou

# 打开文件以保存指标
with open("resnet18-mnist-la.txt", "w") as f:
    # 训练模型
    for epoch in range(100):  # 训练多个 epoch
        running_loss = 0.0
        running_iou = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data

            # 零梯度
            def closure():
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                return loss

            optimizer.step(closure)
            loss = closure().item()
            running_loss += loss

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
