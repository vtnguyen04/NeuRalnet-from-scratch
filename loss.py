import numpy as np
from autograd import AutogradContext
import scipy.sparse
from typing import List, Tuple
from abc import ABC, abstractmethod

class Loss(ABC):

    def __init__(self):
        self.inputs = None

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.forward(y_pred, y_true)

    def propagate_backward(self, grad: np.ndarray) -> None:
        model = AutogradContext.get_model()
        for module in reversed(model._children.values()):
            grad = module.backward(grad)
 
class MSE(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.inputs = (y_pred, y_true)
        return 1 / 2 * np.mean((y_pred - y_true) ** 2)

    def backward(self) -> None:
        y_pred, y_true = self.inputs
        grad = (y_pred - y_true) / y_true.shape[0]
        self.propagate_backward(grad)


class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis = 1, keepdims = True))
        return exp_x / np.sum(exp_x, axis = 1, keepdims = True)

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        
        y_pred = self.softmax(logits)
        self.inputs = (y_pred, y_true)

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if scipy.sparse.issparse(y_true):
            y_true = y_true.toarray()
            
        loss = -np.sum(y_true * np.log(y_pred), axis = 1)
        return np.mean(loss)

    def backward(self) -> None:
        y_pred, y_true = self.inputs
                
        if scipy.sparse.issparse(y_true):
            y_true = y_true.toarray()
        
        grad = y_pred - y_true
        
        self.propagate_backward(grad)


