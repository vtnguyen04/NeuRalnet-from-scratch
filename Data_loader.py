import numpy as np
import struct
from array import array
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any
# '''
# Define a class for loading the MNIST dataset from the files
# Source: https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
# '''
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        '''
        Set the files path
        TRAIN_IMAGES_FILE_PATH: Path to the train images
        training_labels_filepath: Path to the train labels
        test_images_filepath: Path to the test images
        test_labels_filepath: Path to the test labels
        '''
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):
        '''
        Read the images and labels
        images_filepath: Path to the images
        labels_filepath: Path to the labels
        Return: The numpy array containing the images and labels
        '''        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28 *28)
            images[i][:] = img            
        
        return np.array(images), np.array(labels)
            
    def load_data(self):
        '''
        Load the train and test images, train and test labels from the file paths
        '''
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)
    

# Define the paths of the files
INPUT_PATH = 'Mnist'
TRAIN_IMAGES_FILE_PATH = os.path.join(INPUT_PATH, 'train-images.idx3-ubyte')
TRAIN_LABELS_FILE_PATH = os.path.join(INPUT_PATH, 'train-labels.idx1-ubyte')
TEST_IMAGES_FILE_PATH = os.path.join(INPUT_PATH, 'test-images.idx3-ubyte')
TEST_LABELS_FILE_PATH = os.path.join(INPUT_PATH, 'test-labels.idx1-ubyte')

# Create MNISTDataloader and load the images
mnist = MnistDataloader(
    training_images_filepath=TRAIN_IMAGES_FILE_PATH,
    training_labels_filepath=TRAIN_LABELS_FILE_PATH,
    test_images_filepath=TEST_IMAGES_FILE_PATH,
    test_labels_filepath=TEST_LABELS_FILE_PATH
)

# Split into train and test set
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Split into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)

# Standardize the train, validation and test dataset
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

class DataLoader:
    def __init__(self, X, y, batch_size = 32, shuffle = True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.indices = np.arange(self.n_samples)
    
    def __iter__(self):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.index >= self.n_samples:
            raise StopIteration

        batch_indices = self.indices[self.index: self.index + self.batch_size]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]

        self.index += self.batch_size
        return (batch_X, batch_y)
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):   
        if isinstance(idx, slice):
            X_slice = self.X[self.indices[idx]]
            y_slice = self.y[self.indices[idx]]
            return X_slice, y_slice
        elif isinstance(idx, int):
            if idx < 0:  
                idx += self.n_samples
            if idx >= self.n_samples:
                raise IndexError("index out of dataset")
            return (self.X[self.indices[idx]], self.y[self.indices[idx]])
        elif isinstance(idx, list) or isinstance(idx, np.ndarray):
            X_subset = self.X[self.indices[idx]]
            y_subset = self.y[self.indices[idx]]
            return X_subset, y_subset
        else:
            raise TypeError("Not valid input type")
        
