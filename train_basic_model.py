import numpy as np
from loss import CrossEntropy
from Optimizers import Adam
from Module import Module
from activation import GELU, ReLU
from Layers import Linear, Sequential, Dropout, BatchNormalization
from Data_loader import X_train, y_train, X_val, y_val, X_test, y_test
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib
matplotlib.use('Qt5Agg') 

encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_val_encoded = encoder.transform(y_val.reshape(-1, 1)).toarray()
y_test_encoded = encoder.transform(y_test.reshape(-1, 1)).toarray()

model = Sequential(
            Linear(784, 256),
            BatchNormalization(256),
            GELU(),
            Dropout(0.3),
            Linear(256, 128),
            BatchNormalization(128),
            GELU(),
            Dropout(0.3),
            Linear(128, 64),
            BatchNormalization(64),
            GELU(),
            Dropout(0.3),
            Linear(64, 10)
        )
model.summary()

loss_fn = CrossEntropy()

optimizer = Adam(model.parameters(), learning_rate = 0.0005)

def save_model(filepath, model: Module) -> None:
    with open(filepath, 'wb') as f:
        pickle.dump({'model': model,}, f)
    tqdm.write(f"Model saved to {filepath}")


path = os.path.join(os.getcwd(), 'model.pkl')

def train_fn(num_epochs: int, model: Module):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        model.train()
        outputs = model(X_train)
        
        loss = loss_fn(outputs, y_train_encoded)
        loss_fn.backward()
        model.eval()
        predicted = model(X_val)
        optimizer.step()
        accuracy = np.mean(np.argmax(predicted, axis = 1) == np.argmax(y_val_encoded, axis = 1))
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
    
    # save_model(path, model)


# train_fn(100, model)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    return model

model = load_model(path)
indices = np.random.randint(0, 1000, (10, ))
X = X_test[indices]
test_accuracy = np.mean(np.argmax(model(X_test), axis = 1) == np.argmax(y_test_encoded, axis = 1))
print(f'test-accuracy = {test_accuracy}')

num_samples = 10
random_indices = np.random.choice(len(X_test), num_samples, replace = False)

fig, axes = plt.subplots(2, 5, figsize = (15, 6))
axes = axes.ravel()

def predict_single_image(model, image):
    # Reshape image to (1, 784) as the model expects a batch
    image = image.reshape(1, -1)
    model.eval()
    prediction = model(image)
    predicted_class = np.argmax(prediction, axis = 1)
    return predicted_class

for i, idx in enumerate(random_indices):
    image = X_test[idx].reshape(28, 28)
    true_label = np.argmax(y_test_encoded[idx])
    predicted_label = predict_single_image(model, image)
    axes[i].imshow(image, cmap = 'gray')
    axes[i].set_title(f"True: {true_label}, Pred: {predicted_label}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()
plt.close(fig)  

class NN(Module):
    def __init__(self, in_features: int, out_feature: int):
        super(NN, self).__init__()
        
        self.linear1 = Linear(in_features, 64)
        self.relu1 = ReLU()
        self.drop_out = Dropout(0.3)
        self.linear2 = Linear(64, 10)
        self.relu2 = ReLU()
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        x = self.linear1(inputs)
        x = self.relu1(x)
        x = self.drop_out(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x

model2 = NN(784, 10)
model2.summary()