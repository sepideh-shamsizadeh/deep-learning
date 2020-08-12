import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from lr_utils import load_dataset
from framework import ReLU, sigmoid, derivation_of_ReLU


# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Many software bugs in deep learning come from having matrix/vector dimensions that don't fit. If you can keep your
# matrix/vector dimensions straight you will go a long way toward eliminating many bugs.
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# For convenience, you should now reshape images of shape (num_px, num_px, 3) in a numpy-array of shape (num_px  ∗∗
# num_px  ∗∗  3, 1). After this, our training (and test) dataset is a numpy-array where each column represents a
# flattened image. There should be m_train (respectively m_test) columns.
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the
# pixel value is actually a vector of three numbers ranging from 0 to 255.
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# Initialization
n1 = 5
n2 = 10
nx = train_set_x.shape[0]
W1 = np.random.randn(n1, nx)*np.sqrt(2/n1)
W2 = np.random.randn(n2, n1)*np.sqrt(2/n1)
b1 = 0
b2 = 0
alpha = 0.01

# Train model
for i in range(10):
    # Forward propagation
    Z1 = np.dot(W1, train_set_x) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    # Backward propagation
    dZ2 = A2 - train_set_y
    dW2 = (1/m_train) * np.dot(dZ2, A1.T)
    db2 = (1/m_train) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(dW2.T, dZ2) * derivation_of_ReLU(Z1)
    dW1 = (1/m_train) * np.dot(dZ1, train_set_x.T)
    db1 = (1 / m_train) * np.sum(dZ1, axis=1, keepdims=True)
    J = 1/m_train * np.sum(train_set_y*np.log(A2) + (1-train_set_y)*np.log(1-A2))
    print(J)
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2



