import numpy as np
from lr_utils import load_dataset
from framework import sigmoid


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
nh = 290
ny = train_set_y.shape[1]
nx = train_set_x.shape[0]
W1 = np.random.randn(nh, nx) * 1.8
W2 = np.random.randn(ny, nh) * 1.8
b1 = np.zeros(shape=(nh, 1))
b2 = np.zeros(shape=(ny, 1))
alpha = 0.06

# Train model
for i in range(2000):
    # Forward propagation
    Z1 = np.dot(W1, train_set_x) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    # Backward propagation
    dZ2 = A2 - train_set_y
    dW2 = (1/m_train) * np.matmul(dZ2, A1.T)
    db2 = (1/m_train) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = 1 - np.tanh(Z1) ** 2
    dW1 = (1/m_train) * np.matmul(dZ1, train_set_x.T)
    db1 = (1 / m_train) * np.sum(dZ1, axis=1, keepdims=True)
    cost = np.sum(np.log(A2)*train_set_y + (1 - train_set_y)* np.log(1 - A2)) / m_train
    cost = float(np.squeeze(cost))
    if i % 200 == 0:
        print("Cost after iteration %i: %f" % (i, cost))
        Y_prediction_train = A2 > 0.5
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

# Predict
Z1 = np.dot(W1, test_set_x) + b1
A1 = np.tanh(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)
Y_prediction_test = A2 > 0.5
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))




