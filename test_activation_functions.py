import pytest
import numpy as np
from framework import sigmoid, ReLU, tanh, leaky_ReLU


def test_sigmoid():
    assert 0.9820137900379085 == sigmoid(4)
    assert np.all(
        np.array(
            [0.8807970779778823, 0.9525741268224334, 0.9820137900379085, 0.9933071490757153]
        ) == sigmoid(np.array([2, 3, 4, 5]))
    )


def test_tangh():
    assert 0.999909204262595 == tanh(5)
    assert np.all(
        np.array(
            [0.0, 0.7615941559557649, -0.7615941559557649, 0.9999999958776926]
        ) == tanh(np.array([0, 1, -1, 10]))
                  )