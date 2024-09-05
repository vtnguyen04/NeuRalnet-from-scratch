#%%
import numpy as np
from loss import CrossEntropy, MSE
from Optimizers import Adam, cosine_annealing, step_decay
from activation import GELU, ReLU, Sigmoid
from Layers import Linear, Sequential, Dropout, BatchNormalization
from Data_loader import X_train, y_train, X_val, y_val, X_test, y_test, DataLoader
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib
from autograd import AutogradContext
matplotlib.use('Qt5Agg')

from pl import LightningModule, Trainer
from callbacks import ProgressLogger, ModelCheckpoint, EarlyStopping, LearningRateScheduler

encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_val_encoded = encoder.transform(y_val.reshape(-1, 1)).toarray()
y_test_encoded = encoder.transform(y_test.reshape(-1, 1)).toarray()

train_loader = DataLoader(X_train, y_train_encoded, batch_size = 512, shuffle = True)
val_loader = DataLoader(X_val, y_val_encoded, batch_size = 512, shuffle = False)
test_loader = DataLoader(X_test, y_test_encoded, batch_size = 512, shuffle = False)

class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = Sequential(
            Linear(784, 256),
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
            Linear(64, 10)
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

model = MNISTModel()
model.summary()

# Callbacks
progress_logger = ProgressLogger()
model_checkpoint = ModelCheckpoint('best_model.pkl', monitor = 'val_accuracy', mode = 'max', save_best_only = True)
early_stopping = EarlyStopping(monitor = 'val_loss', mode = 'max', patience = 50, min_delta = 0.5)
lr_scheduler = LearningRateScheduler(cosine_annealing(initial_lr = 0.0001, T_max = 50))

trainer = Trainer(model, max_epochs = 50, callbacks = [progress_logger, model_checkpoint, early_stopping, lr_scheduler])

trainer.fit(train_loader, val_loader)

test_loss, test_accuracy = trainer.test(test_loader)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


def predict_single_image(model, image):
    model.eval()
    prediction = model(image.reshape(1, -1))
    return prediction.argmax(axis = 1)

num_samples = 10
random_indices = np.random.choice(len(X_test), num_samples, replace = False)

fig, axes = plt.subplots(2, num_samples // 2, figsize = (15, 6))
axes = axes.ravel()

X, y = test_loader[random_indices]
for idx in range(len(X)):
    image = X[idx]
    true_label = y[idx].argmax()
    predicted_label = predict_single_image(model, image)
    axes[idx].imshow(image.reshape(28, 28), cmap = 'gray')
    axes[idx].set_title(f"True: {true_label}, Pred: {predicted_label}")
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
# %%
