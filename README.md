# Neural Network Framework Inspired by PyTorch & PyTorch Lightning

This repository contains the implementation of a neural network framework that mimics the behavior of PyTorch and PyTorch Lightning. Created by me (Võ Thành Nguyễn), this framework is designed to provide a user-friendly interface for building, training, and validating deep learning models, Help everyone gain a deeper understanding of the implementation of an artificial network from scratch.


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

https://github.com/vtnguyen04/NeuRalnet-from-scratch.git

```bash
git clone https://github.com/vtnguyen04/NeuRalnet-from-scratch.git
cd NeuRalnet-from-scratch
```

## Usage

Here is a simple example to demonstrate how to define a model, train, and evaluate it using the framework:

```python

# Author: Vo Thanh Nguyen
# Date Created: 2024-09-05
# Description: This script demonstrates demonstrate how to define a model, train, and evaluate it using the framework:

import numpy as np
from loss import CrossEntropy
from Optimizers import Adam
from activation import ReLU
from Layers import Linear, Sequential, Dropout, BatchNormalization
from Data_loader import DataLoader
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib
from autograd import AutogradContext
matplotlib.use('Qt5Agg')

from pl import LightningModule, Trainer
from callbacks import ProgressLogger, ModelCheckpoint, EarlyStopping, LearningRateScheduler

""" setup your data here """
num_inputs, num_feature_inputs, num_feature_outputs = 10, 10, 10

X_train = np.random.rand(num_inputs, num_feature_inputs)
y_train = np.random.rand(num_inputs, num_feature_outputs)

X_val = np.random.rand(num_inputs, num_feature_inputs)
y_val = np.random.rand(num_inputs, num_feature_outputs)

train_loader = DataLoader(X_train, y_train, batch_size = 512, shuffle = True)
val_loader = DataLoader(X_val, y_val, batch_size = 512, shuffle = True)

class NN(LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = Sequential(
            Linear(num_feature_inputs, 256),
            BatchNormalization(256),
            ReLU(),
            Dropout(0.5),
            Linear(256, 128),
            BatchNormalization(128),
            ReLU(),
            Dropout(0.3),
            Linear(128, 64),
            BatchNormalization(64),
            GELU(),
            Linear(64, num_feature_outputs)
        )
        self.loss_fn = CrossEntropy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        accuracy = (y_pred.argmax(axis = 1) == y.argmax(axis = 1)).mean()
        return loss, accuracy
    
    def configure_optimizers(self):
        return Adam(self.parameters(), learning_rate = 0.0001)

model.summary()
# Example usage
model = NN()
model.summary()
trainer = Trainer(model)
trainer.fit(train_loader, val_loader)
```


