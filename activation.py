import numpy as np
from Module import Module

class GELU(Module):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return 0.5 * inputs * (1 + np.tanh(np.sqrt(2 / np.pi) * (inputs + 0.044715 * np.power(inputs, 3))))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.inputs + 0.044715 * np.power(self.inputs, 3))))
        pdf = np.exp(-(self.inputs ** 2) / 2) / np.sqrt(2 * np.pi)
        return grad * (cdf + self.inputs * pdf)

class ReLU(Module):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (self.inputs > 0)
    
class Sigmoid(Module):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.outputs * (1 - self.outputs)