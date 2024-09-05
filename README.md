# Neural Network Framework Inspired by PyTorch & PyTorch Lightning

This repository contains the implementation of a neural network framework that mimics the behavior of PyTorch and PyTorch Lightning. Created by Võ Thành Nguyễn, this framework is designed to provide a user-friendly interface for building, training, and validating deep learning models, with a focus on customizability and extensibility.

## Features

- **Module System**: Similar to PyTorch, this framework offers a base `Module` class from which all layers and models inherit, facilitating the creation of complex neural networks through a simple and intuitive API.
- **Automatic Differentiation**: Equipped with a custom `Autograd` system that allows for automatic computation of gradients, making it easier to implement backpropagation without manual gradient calculations.
- **Optimization Algorithms**: Includes implementations of popular optimizers like SGD and Adam, complete with adjustable learning rates and other hyperparameters for robust training routines.
- **Loss Functions**: A variety of loss functions are available to cater to different kinds of problems, from regression to classification.
- **Training and Validation Loops**: Inspired by PyTorch Lightning, the framework simplifies the training process with predefined training, validation, and testing loops, while still allowing for extensive customization through callbacks and hooks.
- **Custom Callbacks**: Support for custom callbacks to extend functionality during training, such as learning rate scheduling, early stopping, and logging metrics.

## Structure

The framework is organized into several key files, each responsible for different aspects of the modeling and training process:

- **`Module.py`**: Defines the base `Module` class for all models and layers.
- **`Layers.py`**: Contains implementations of various neural network layers like Linear, Convolutional, etc.
- **`activation.py`**: Implements common activation functions.
- **`autograd.py`**: Provides the automatic differentiation capabilities via the `AutogradContext` class.
- **`loss.py`**: Includes definitions for different loss functions.
- **`Optimizers.py`**: Contains classes for various optimization algorithms.
- **`pl.py`**: Mimics PyTorch Lightning's high-level interface for training and validation.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/PathToRepository/NeuralNetworkFramework.git
cd NeuralNetworkFramework
```

## Usage

Here is a simple example to demonstrate how to define a model, train, and evaluate it using the framework:

```python
from Module import Linear, Activation
from pl import LightningModule, Trainer
from Optimizers import Adam
from loss import CrossEntropyLoss

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(784, 128)
        self.activation = Activation('relu')
        self.layer2 = Linear(128, 10)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        return self.layer2(x)

    def training_step(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        return loss

    def configure_optimizers():
        return Adam(self.parameters())

# Example usage
model = MyModel()
trainer = Trainer(model)
trainer.fit(train_loader, val_loader)
```

## Contribution

Contributions are welcome! If you'd like to improve the framework or add new features, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

This README provides a comprehensive guide to getting started with the framework, from installation to defining and training models.
