#制图
import matplotlib.pyplot as plt
import re


def parse_file(filename):
    epochs = []
    losses = []
    ious = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.match(r'Epoch: (\d+), Average Loss: ([\d.]+), Average IOU: ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                iou = float(match.group(3))
                epochs.append(epoch)
                losses.append(loss)
                ious.append(iou)

    return epochs, losses, ious


def plot_metrics(filenames):
    all_epochs = []
    all_losses = []
    all_ious = []

    for filename in filenames:
        epochs, losses, ious = parse_file(filename)
        all_epochs.append(epochs)
        all_losses.append(losses)
        all_ious.append(ious)

    #  Loss
    plt.figure(figsize=(12, 6))
    for i in range(len(filenames)):
        plt.plot(all_epochs[i], all_losses[i], label=filenames[i].split('/')[-1].replace('.txt', ''))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('./loss_MNIST_epochs.png')
    plt.close()

    #  IOU
    plt.figure(figsize=(12, 6))
    for i in range(len(filenames)):
        plt.plot(all_epochs[i], all_ious[i], label=filenames[i].split('/')[-1].replace('.txt', ''))
    plt.xlabel('Epoch')
    plt.ylabel('IOU')
    plt.title('IOU over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('./iou_MNIST_epochs.png')
    plt.close()


# Example usage
filenames = [
    './ResNet_MNIST_LA.txt',
    './CNN_MNIST_LA.txt',
    './ENet_MNIST_LA.txt',
    './ResNet_MNIST_ADAM.txt',
    './ENet_MNIST_ADAM.txt',
    './CNN_MNIST_ADAM.txt',
    # './ResNet_MNIST.txt',
    # './ENet_MNIST.txt',
    # './CNN_MNIST.txt'
]
plot_metrics(filenames)

