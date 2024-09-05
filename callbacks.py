import numpy as np
import pickle
from typing import Literal
from abc import ABC, abstractmethod
from tqdm import tqdm

class Callback(ABC):
    @abstractmethod
    def on_train_start(self, trainer, model):
        pass

    @abstractmethod
    def on_epoch_start(self, trainer, model):
        pass

    @abstractmethod
    def on_batch_start(self, trainer, model, batch, batch_idx):
        pass

    def on_batch_end(self, trainer, model, batch, batch_idx):
        pass

    @abstractmethod
    def on_epoch_end(self, trainer, model):
        pass

    @abstractmethod
    def on_train_end(self, trainer, model):
        pass

class ProgressLogger(Callback):
    def __init__(self):
        self.train_pbar = None
        self.epoch_pbar = None

    def on_train_start(self, trainer, model):
        self.train_pbar = tqdm(total = trainer.max_epochs, desc = "Training", colour = 'cyan')

    def on_epoch_start(self, trainer, model):
        self.epoch_pbar = tqdm(total = len(trainer.train_loader) // 2048, 
                               desc = f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
                                colour = 'green')

    def on_batch_end(self, trainer, model, batch, batch_idx):
        self.epoch_pbar.update(1)

    def on_epoch_end(self, trainer, model):
        self.epoch_pbar.close()
        self.train_pbar.update(1)
        tqdm.write(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} completed. \
                   Val Loss: {trainer.validation_metrics.get('val_loss', 'N/A'):.4f} \
                   Val Accuracy: {trainer.validation_metrics.get('val_acc', 'N/A'):.4f}")

    def on_train_end(self, trainer, model):
        self.train_pbar.close()

    def on_batch_start(self, trainer, model, batch, batch_idx):
        pass

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor : Literal['val_loss', 'val_accuracy'],  mode : Literal['min', 'max'], save_best_only = True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, trainer, model):
        current_score = trainer.validation_metrics.get(self.monitor)
        if current_score is None:
            return

        if self.mode == 'min':
            improved = current_score < self.best_score
        else:
            improved = current_score > self.best_score

        if improved or not self.save_best_only:
            self.best_score = current_score
            self.save_model(model)
            tqdm.write(f"Model saved to {self.filepath}. Best {self.monitor}: {self.best_score:.4f}")

    def save_model(self, model):
        with open(self.filepath, 'wb') as f:
            pickle.dump({'model': model}, f)

    def on_train_start(self, trainer, model):
        pass

    def on_epoch_start(self, trainer, model):
        pass

    def on_batch_start(self, trainer, model, batch, batch_idx):
        pass

    def on_batch_end(self, trainer, model, batch, batch_idx):
        pass

    def on_train_end(self, trainer, model):
        pass


class EarlyStopping(Callback):
    def __init__(self, monitor : Literal['val_loss', 'val_accuracy'], mode : Literal['min', 'max'], patience = 5, min_delta = 0.0):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer, model):
        current_score = trainer.validation_metrics.get(self.monitor)
        if current_score is None:
            return

        if self.mode == 'min':
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.stop_training = True
                tqdm.write(f"Early stopping triggered at epoch {self.stopped_epoch + 1}")

    def on_train_start(self, trainer, model):
        pass

    def on_epoch_start(self, trainer, model):
        pass

    def on_batch_start(self, trainer, model, batch, batch_idx):
        pass

    def on_batch_end(self, trainer, model, batch, batch_idx):
        pass

    def on_train_end(self, trainer, model):
        pass


class LearningRateScheduler(Callback):
    def __init__(self, schedule_fn):
        self.schedule_fn = schedule_fn

    def on_epoch_start(self, trainer, model):
        optimizer = trainer.optimizer
        current_lr = optimizer.learning_rate
        new_lr = self.schedule_fn(trainer.current_epoch, current_lr)
        optimizer.learning_rate = new_lr
        tqdm.write(f"Epoch {trainer.current_epoch}: Learning rate set to {new_lr}")

    def on_epoch_end(self, trainer, model):
        pass

    def on_train_start(self, trainer, model):
        pass

    def on_batch_start(self, trainer, model, batch, batch_idx):
        pass

    def on_batch_end(self, trainer, model, batch, batch_idx):
        pass

    def on_train_end(self, trainer, model):
        pass