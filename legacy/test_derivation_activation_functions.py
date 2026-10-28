import pytest
import numpy as np
from framework import derivation_of_sigmoid, derivation_of_leaky_ReLU, derivation_of_ReLU, derivation_of_tanh


def test_derivation_of_sigmoid():
    assert 0.017865830940122396 == derivation_of_sigmoid(5)
    assert np.all(
        np.array(
            [0.23688281808991007, 0.36552928931500245, 0.37937142700199805, 0.017865830940122396, 0.3344179575849439]
        ) == derivation_of_sigmoid(np.array([-1, 0, 0.2, 5, -0.3]))
    )


def test_derivation_of_tanh():
    assert 0.00018158323094408235 == derivation_of_tanh(5)
    assert np.all(
        np.array(
            [0.41997434161402614, 1.0, 0.9610429829661166, 0.00018158323094408235, 0.9151369618266292]
        ) == derivation_of_tanh(np.array([-1, 0, 0.2, 5, -0.3]))
                  )


def test_derivation_of_Relu():
    assert [1] == derivation_of_ReLU([7])
    assert np.all(
        [1, 0, 1, 1, 1, 0, 1, 1] == derivation_of_ReLU(
            np.array([0, -3, 9, 0, 1.1, -0.1, 0, 1.2])
        )
    )


def test_derivation_of_leaky_ReLU():
    assert [1] == derivation_of_leaky_ReLU([7])
    assert np.all(
        [1, 0.01, 1, 1, 1, 0.01, 1, 1] == derivation_of_leaky_ReLU(
            np.array([0, -3, 9, 0, 1.1, -0.1, 0, 1.2])
        )
    )

