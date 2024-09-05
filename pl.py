from Module import Module
import numpy as np
from typing import List, Tuple
from abc import abstractmethod
from callbacks import Callback

class LightningModule(Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = None

    @abstractmethod
    def training_step(self, batch: Tuple[np.ndarray, np.ndarray]):
        raise NotImplementedError
    
    @abstractmethod
    def validation_step(self, batch: Tuple[np.ndarray, np.ndarray]):
        raise NotImplementedError
    
    def test_step(self, batch: Tuple[np.ndarray, np.ndarray]):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        accuracy = (outputs.argmax(axis = 1) == targets.argmax(axis = 1)).mean()
        return loss, accuracy
    
    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError


class Trainer:
    def __init__(self, model: LightningModule, max_epochs: int = 10, callbacks: List[Callback] = None):
        self.model = model
        self.max_epochs = max_epochs
        self.callbacks = callbacks or []
        self.current_epoch = 0 # use for learning rate schedule 
        self.stop_training = False # EarlyStopping
        self.validation_metrics = {} # save validation metrics
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None 

    def _invoke_callbacks(self, callback_method: str, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, callback_method, None) 
            if method:
                method(self, *args, **kwargs)

    def fit(self, train_loader, val_loader = None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.model.configure_optimizers()  

        self._invoke_callbacks('on_train_start', self.model)

        for self.current_epoch in range(self.max_epochs):
            if self.stop_training:
                break

            self._invoke_callbacks('on_epoch_start', self.model)
            self.model.train()

            for batch_idx, batch in enumerate(train_loader):
                self._invoke_callbacks('on_batch_start', self.model, batch, batch_idx)
                
                loss = self.model.training_step(batch)
                self.optimizer.zero_grad()
                self.model.loss_fn.backward()
                self.optimizer.step()

                self._invoke_callbacks('on_batch_end', self.model, batch, batch_idx)

            if val_loader is not None:
                self.model.eval()
                val_loss = []
                val_accuracy = []
                for batch_idx, batch in enumerate(val_loader):
                    loss, accuracy = self.model.validation_step(batch)
                    val_loss.append(loss)
                    val_accuracy.append(accuracy)

                avg_val_loss = np.mean(val_loss)
                avg_val_accuracy = np.mean(val_accuracy)
                self.validation_metrics['val_loss'] = avg_val_loss
                self.validation_metrics['val_acc'] = avg_val_accuracy

            self._invoke_callbacks('on_epoch_end', self.model)

        self._invoke_callbacks('on_train_end', self.model)

    def test(self, test_loader):
        self.model.eval()
        test_loss = []
        test_accuracy = []
        for batch in test_loader:
            loss, accuracy = self.model.test_step(batch)
            test_loss.append(loss)
            test_accuracy.append(accuracy)
        avg_test_loss = np.mean(test_loss)
        avg_test_accuracy = np.mean(test_accuracy)
        return avg_test_loss, avg_test_accuracy