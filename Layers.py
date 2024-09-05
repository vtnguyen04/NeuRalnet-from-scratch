import numpy as np
from Module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self._parameters['weights'] = np.random.randn(in_features, out_features) * np.sqrt(2 / (2 * in_features + 1))
        self._parameters['bias'] = np.random.randn(1, out_features) * np.sqrt(1 / out_features)
        self._gradients['weights'] = np.zeros_like(self._parameters['weights'])
        self._gradients['bias'] = np.zeros_like(self._parameters['bias'])

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.dot(inputs, self._parameters['weights']) + self._parameters['bias']

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self._gradients['weights'] += np.dot(self.inputs.T, grad)
        self._gradients['bias'] += np.sum(grad, axis = 0, keepdims = True)
        return np.dot(grad, self._parameters['weights'].T)

class Dropout(Module):
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size = inputs.shape) / (1 - self. dropout_rate)
            return inputs * self.mask
        return inputs

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.training:
            return grad * self.mask
        return grad

class BatchNormalization(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(BatchNormalization, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self._parameters['gamma'] = np.ones((1, num_features))
        self._parameters['beta'] = np.zeros((1, num_features))
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        self._gradients['gamma'] = np.zeros_like(self._parameters['gamma'])
        self._gradients['beta'] = np.zeros_like(self._parameters['beta'])
        self.inputs = None
        self.x_normalized = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.training:
            mean = np.mean(inputs, axis=0, keepdims=True)
            var = np.var(inputs, axis=0, keepdims=True)
            self.x_normalized = (inputs - mean) / np.sqrt(var + self.eps)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            self.x_normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self._parameters['gamma'] * self.x_normalized + self._parameters['beta']

    def backward(self, grad: np.ndarray) -> np.ndarray:
        N, D = grad.shape
        self._gradients['gamma'] = np.sum(grad * self.x_normalized, axis=0, keepdims=True)
        self._gradients['beta'] = np.sum(grad, axis=0, keepdims=True)

        dx_normalized = grad * self._parameters['gamma']
        if self.training:
            var = np.var(self.x_normalized, axis=0, keepdims=True)
            dvar = np.sum(dx_normalized * (self.x_normalized) * -0.5 * (var + self.eps) ** (-3/2), axis=0, keepdims=True)
            dmean = np.sum(dx_normalized * -1 / np.sqrt(var + self.eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (self.x_normalized * np.sqrt(var + self.eps)), axis=0, keepdims=True)
            return dx_normalized / np.sqrt(var + self.eps) + dvar * 2 * (self.x_normalized * np.sqrt(var + self.eps)) / N + dmean / N
        else:
            return dx_normalized / np.sqrt(self.running_var + self.eps)


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for module in modules:
            self.add_module(module)

    def add_module(self, module: Module):
        name = type(module).__name__
        base_name = name
        index = 1
        while name in self._children:
            name = f"{base_name}_{index}"
            index += 1
        self.__setattr__(name, module)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for module in self._children.values():
            inputs = module(inputs)
        return inputs

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