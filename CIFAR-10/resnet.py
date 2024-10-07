import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.models import resnet18


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = resnet18(pretrained=False)
net.fc = nn.Linear(net.fc.in_features, 10)  


criterion = nn.CrossEntropyLoss()


with open("resnet18-cifar-no-optimizer.txt", "w") as f:
 
    for epoch in range(100):  
        running_loss = 0.0
        running_iou = 0.0
        for i, data in enumerate(trainloader, 0):
          
            inputs, labels = data

           
            net.zero_grad()

            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            
            with torch.no_grad():
                for param in net.parameters():
                    if param.grad is not None:
                        param -= 0.001 * param.grad  

            running_loss += loss.item()

           
            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                intersection = (predicted == labels).sum().item()
                union = len(labels)
                iou = intersection / union
                running_iou += iou

        
        epoch_loss = running_loss / len(trainloader)
        epoch_iou = running_iou / len(trainloader)
        f.write(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average IOU: {epoch_iou:.4f}\n')
        print(f'Epoch: {epoch + 1}, Average Loss: {epoch_loss:.6f}, Average IOU: {epoch_iou:.4f}')

    print('Finished Training')
