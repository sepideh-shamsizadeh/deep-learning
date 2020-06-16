import pytest
import numpy as np
from framework import sigmoid, ReLU, tanh, leaky_ReLU


def test_sigmoid():
    assert 0.9820137900379085 == sigmoid(4)
    assert np.all(np.array(
        [0.8807970779778823, 0.9525741268224334, 0.9820137900379085, 0.9933071490757153]
    ) == sigmoid(np.array([2, 3, 4, 5])))
