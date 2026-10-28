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


def test_tanh():
    assert 0.999909204262595 == tanh(5)
    assert np.all(
        np.array(
            [0.0, 0.7615941559557649, -0.7615941559557649, 0.9999999958776926]
        ) == tanh(np.array([0, 1, -1, 10]))
                  )


def test_Relu():
    assert 7 == ReLU(7)
    assert np.all(
        [0, 3, 9, 0, 1.1, 0.1, 0, 1.2] == ReLU(
            np.array([-1, 3, 9, -0.4, 1.1, 0.1, -3, 1.2])
        )
    )


def test_leaky_ReLU():
    assert 7 == leaky_ReLU(7)
    assert np.all(
        [-0.01, 3, 9, -0.004, 1.1, 0.1, -0.03, 1.2] == leaky_ReLU(
            np.array([-1, 3, 9, -0.4, 1.1, 0.1, -3, 1.2])
        )
    )

