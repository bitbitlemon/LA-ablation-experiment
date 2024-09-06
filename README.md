
# LA Ablation Experiment

## Introduction

This repository contains the implementation of experiments conducted for **Loss-Agnostic Optimization Ablation** across different architectures such as CNN, ENet, and ResNet on popular datasets like CIFAR-10 and MNIST. The aim of this project is to compare the performance of different optimization methods (e.g., Adam, LA, and custom variants like LBFGSAdam) in training neural networks and analyzing their effects on model convergence, accuracy, and loss minimization.

The experiments investigate:
- **Ablation Studies**: Understanding the contributions of different optimizers on neural network training.
- **Performance Benchmarks**: Comparing networks trained on CIFAR-10 and MNIST using CNN, ENet, and ResNet architectures.
- **Optimizer Comparison**: Evaluation of standard optimizers (Adam) versus custom approaches (LA, LBFGSAdam).

## Project Structure

The project is organized as follows:

```
LA-ablation-experiment-main/
│
├── CIFAR-10/
│   ├── cnn.py                # CNN architecture implementation for CIFAR-10
│   ├── cnn_adam.py           # CNN + Adam optimizer
│   ├── cnn_la.py             # CNN + LA optimizer
│   ├── enet.py               # ENet architecture for CIFAR-10
│   ├── enet_adam.py          # ENet + Adam optimizer
│   ├── enet_la.py            # ENet + LA optimizer
│   ├── resnet.py             # ResNet architecture for CIFAR-10
│   ├── resnet_adam.py        # ResNet + Adam optimizer
│   ├── resnet_la.py          # ResNet + LA optimizer
│   └── z_plot-10.py          # Plotting results for CIFAR-10 experiments
│
├── MNIST/
│   ├── cnn.py                # CNN architecture implementation for MNIST
│   ├── cnn_adam.py           # CNN + Adam optimizer
│   ├── cnn_la.py             # CNN + LA optimizer
│   ├── enet.py               # ENet architecture for MNIST
│   ├── enet_adam.py          # ENet + Adam optimizer
│   ├── enet_la.py            # ENet + LA optimizer
│   ├── resnet.py             # ResNet architecture for MNIST
│   ├── resnet_adam.py        # ResNet + Adam optimizer
│   ├── resnet_la.py          # ResNet + LA optimizer
│   └── z_plot.py             # Plotting results for MNIST experiments
│
├── LBFGSAdam.py              # Custom optimizer combining LBFGS and Adam
├── .gitattributes            # Git attributes file for repo configuration
```

## Installation

### Dependencies

The project relies on the following dependencies, which can be installed via pip:

```bash
pip install torch torchvision matplotlib numpy
```

Ensure you have Python 3.7 or higher installed. The project also uses PyTorch for deep learning tasks.

### Dataset Preparation

The CIFAR-10 and MNIST datasets are automatically downloaded and handled by PyTorch's `torchvision` package during runtime. No manual dataset preparation is required.

## Usage Instructions

### Running Experiments

Each folder (`CIFAR-10` and `MNIST`) contains scripts for training different architectures using various optimizers. You can run the training for each model by executing the corresponding Python script. For example:

```bash
python CIFAR-10 dataset/cnn_adam.py
```

This will train a CNN on CIFAR-10 using the Adam optimizer. Similarly, you can run experiments with different models (e.g., ResNet, ENet) and optimizers (e.g., LA, LBFGSAdam).

### Custom Optimizer

A custom optimizer combining LBFGS and Adam is provided in the `LBFGSAdam.py` file. You can integrate it into any model by importing the `LBFGSAdam` class and initializing it in place of a standard optimizer.

Example:
```python
from LBFGSAdam import LBFGSAdam

optimizer = LBFGSAdam(model.parameters(), lr=0.01)
```

## Results and Visualization

The results of each experiment are stored in text files (e.g., `CNN_CIFAR.txt`, `ENet_MNIST.txt`) that contain accuracy and loss information for each epoch. You can use the provided plotting scripts (e.g., `z_plot.py`) to visualize the performance of different models and optimizers.

To generate plots for CIFAR-10 experiments, run:

```bash
python CIFAR-10/z_plot-10.py
```

For MNIST experiments:

```bash
python MNIST/z_plot.py
```

The plots will be saved as image files in the project directory.

## Experiment Details

### Optimizers

- **Adam**: A standard stochastic gradient descent method that adapts the learning rate for each parameter.
- **LA**: Loss-Agnostic optimization strategy aiming to improve the generalization ability of models.
- **LBFGSAdam**: A custom hybrid optimizer that combines the memory-efficient LBFGS with Adam's adaptive learning rates for improved convergence.

### Architectures

- **CNN**: A simple convolutional neural network for both CIFAR-10 and MNIST datasets.
- **ENet**: An efficient neural network architecture used for classification tasks.
- **ResNet**: A deep residual network known for its superior performance on complex tasks.

### Datasets

- **CIFAR-10**: A dataset consisting of 60,000 32x32 color images in 10 classes.
- **MNIST**: A dataset of handwritten digits consisting of 60,000 training images and 10,000 test images.

## Conclusion

This project provides an in-depth analysis of different optimization techniques for training deep neural networks on well-known datasets. The ablation study highlights the strengths and weaknesses of each approach, providing valuable insights for researchers and practitioners aiming to optimize neural network training processes.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
