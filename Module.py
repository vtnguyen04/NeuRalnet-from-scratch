import numpy as np
from typing import List, Dict
from autograd import AutogradContext
from abc import ABC, abstractmethod

class Module(ABC):

    def __init__(self):
        self._parameters = {}
        self._gradients = {}
        self._children = {}
        self.training = True
        AutogradContext.set_model(self)

    def __setattr__(self, name, value) -> None:
        if isinstance(value, Module):
            if len(value._children):
                self._children.update(value._children)
            else:
                self._children[name] = value
        return super().__setattr__(name, value)
    
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if len(self._children):
            AutogradContext.set_model(self)
        return self.forward(inputs)
    
    def backward(self, grad: np.ndarray) -> None:
        raise NotImplementedError
    
    def parameters(self) -> List[Dict[str, np.ndarray]]:
        params_with_grads = []
        for name, param in self._parameters.items():
            params_with_grads.append({
                'weights': param,
                'grad': self._gradients[name]
            })
        for module in self._children.values():
            params_with_grads.extend(module.parameters())
        return params_with_grads

    def train(self) -> None:
        self.set_training_mode(True)
    
    def eval(self) -> None:
        self.set_training_mode(False)

    def set_training_mode(self, training: bool) -> None:
        self.training = training
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.set_training_mode(training)

    def summary(self) -> None:
        total_params = 0
        print("Model Summary\n")
        print("Layer (type)               Parameters")
        print("=" * 40)
        for name, module in self._children.items():
            num_params = sum(np.prod(p['weights'].shape) for p in module.parameters())
            total_params += num_params
            print(f"{name:<25} {num_params:,}")
        print("=" * 40)
        print(f"Total Parameters:          {total_params:,}")
