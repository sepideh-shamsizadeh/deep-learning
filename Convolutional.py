import numpy as np


def zero_pad(X, pad):
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))


def conv_single_step(a_slice_prev, W, b):
    return np.sum(a_slice_prev*W)+b


def convolutional():
    return 0