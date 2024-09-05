from typing import List, Dict
import numpy as np
import math

class Optimizer:
    def __init__(self, parameters: List[Dict[str, np.ndarray]], learning_rate: float = 0.001):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            param['grad'].fill(0)

class Adam(Optimizer):
    def __init__(self, parameters: List[Dict[str, np.ndarray]], learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(parameters, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(param['weights']) for param in self.parameters]  # Tích lũy moment bậc 1
        self.v = [np.zeros_like(param['weights']) for param in self.parameters]  # Tích lũy moment bậc 2

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            grad = param['grad']
            
            # Cập nhật moment bậc 1 và bậc 2
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias-corrected moment
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Cập nhật trọng số
            param['weights'] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


# Step decay
step_decay = lambda initial_lr, drop_factor=0.5, epochs_drop=10: \
    lambda epoch, current_lr: initial_lr * (drop_factor ** (epoch // epochs_drop))

# Exponential decay
exponential_decay = lambda initial_lr, decay_rate=0.97: \
    lambda epoch, current_lr: initial_lr * (decay_rate ** epoch)

# Cosine annealing
cosine_annealing = lambda initial_lr, T_max: \
    lambda epoch, current_lr: initial_lr * (1 + math.cos(math.pi * epoch / T_max)) / 2