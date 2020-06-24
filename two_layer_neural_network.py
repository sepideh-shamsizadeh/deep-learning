import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from lr_utils import load_dataset


# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Many software bugs in deep learning come from having matrix/vector dimensions that don't fit. If you can keep your
# matrix/vector dimensions straight you will go a long way toward eliminating many bugs.
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
#
# For convenience, you should now reshape images of shape (num_px, num_px, 3) in a numpy-array of shape (num_px  ∗∗
# num_px  ∗∗  3, 1). After this, our training (and test) dataset is a numpy-array where each column represents a
# flattened image. There should be m_train (respectively m_test) columns.
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the
# pixel value is actually a vector of three numbers ranging from 0 to 255.
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

