
import unittest
import numpy as np
from minidl.tensor import Tensor


class TestTensor(unittest.TestCase):

    def test_tensor_shape(self):
        x = Tensor([1, 2, 3])
        self.assertEqual(x.shape, (3,))

    def test_initial_grad(self):
        x = Tensor([1, 2, 3])
        self.assertIsNone(x.grad)

    def test_scalar_shape(self):
        x = Tensor(5)
        self.assertEqual(x.shape, ())

    def test_matrix_shape(self):
        x = Tensor([[1, 2, 3], [2, 3, 4]])
        self.assertEqual(x.shape, (2, 3))

    def test_default_dtype(self):
        x = Tensor([1, 2, 3])
        self.assertEqual(x.dtype, np.float32)

    def test_custom_dtype(self):
        x = Tensor([1, 2, 3], dtype=np.float64)
        self.assertEqual(x.dtype, np.float64)

    def test_requires_grad(self):
        x = Tensor([1, 2, 3], requires_grad=True)
        self.assertTrue(x.requires_grad)

    def test_default_requires_grad(self):
        x = Tensor([1, 2, 3])
        self.assertFalse(x.requires_grad)

    def test_tensor_data(self):
        x = Tensor([1, 2, 3])
        np.testing.assert_array_equal(
            x.data, np.array([1, 2, 3], dtype=np.float32)
        )

    def test_data_is_numpy_array(self):
        x = Tensor([1, 2, 3])
        self.assertIsInstance(x.data, np.ndarray)

    def test_empty_tensor(self):
        x = Tensor([])
        self.assertEqual(x.shape, (0,))

    def test_tensor_repr(self):
        x = Tensor([1, 2], requires_grad=True)
        representation = repr(x)

        self.assertIn("Tensor(", representation)
        self.assertIn("float32", representation)
        self.assertIn("requires_grad=True", representation)
    
    def test_add_two_tensors(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])

        result = a + b

        self.assertIsInstance(result, Tensor)
        np.testing.assert_array_equal(
            result.data, [5, 7, 9]
        )

    def test_add_scalar(self):
        a = Tensor([1, 2, 3])

        result = a + 10

        np.testing.assert_array_equal(
            result.data, [11, 12, 13]
        )

    def test_add_broadcasting(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = Tensor([10, 20, 30])

        result = a + b

        np.testing.assert_array_equal(
            result.data,
            [[11, 22, 33], [14, 25, 36]]
        )
        self.assertEqual(result.shape, (2, 3))

    def test_add_preserves_dtype(self):
        a = Tensor([1, 2, 3], dtype=np.float64)
        b = Tensor([4, 5, 6], dtype=np.float64)

        result = a + b

        self.assertEqual(result.dtype, np.float64)

    def test_add_does_not_modify_inputs(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])

        result = a + b

        np.testing.assert_array_equal(a.data, [1, 2, 3])
        np.testing.assert_array_equal(b.data, [4, 5, 6])
        np.testing.assert_array_equal(result.data, [5, 7, 9])

    def test_add_incompatible_shapes(self):
        a = Tensor([1, 2, 3])
        b = Tensor([1, 2])

        with self.assertRaises(ValueError):
            _ = a + b

    def test_subtraction(self):
        a = Tensor([5, 6, 7])
        b = Tensor([1, 2, 3])

        result = a - b

        np.testing.assert_array_equal(
            result.data, [4, 4, 4]
        )

    def test_multiplication(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])

        result = a * b

        np.testing.assert_array_equal(
            result.data, [4, 10, 18]
        )

    def test_division(self):
        a = Tensor([10, 20, 30])
        b = Tensor([2, 4, 5])

        result = a / b

        np.testing.assert_allclose(
            result.data, [5, 5, 6]
        )

    def test_negation(self):
        a = Tensor([1, -2, 3])

        result = -a

        np.testing.assert_array_equal(
            result.data, [-1, 2, -3]
        )

    def test_power(self):
        a = Tensor([2, 3, 4])

        result = a ** 2

        np.testing.assert_array_equal(
            result.data, [4, 9, 16]
        )

    def test_multiplication_broadcasting(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = Tensor([2, 3, 4])

        result = a * b

        np.testing.assert_array_equal(
            result.data,
            [[2, 6, 12], [8, 15, 24]]
        )


    def test_reverse_addition(self):
        a = Tensor([1, 2, 3])

        result = 10 + a

        np.testing.assert_array_equal(
            result.data, [11, 12, 13]
        )

    def test_reverse_subtraction(self):
        a = Tensor([1, 2, 3])

        result = 10 - a

        np.testing.assert_array_equal(
            result.data, [9, 8, 7]
        )

    def test_reverse_multiplication(self):
        a = Tensor([1, 2, 3])

        result = 10 * a

        np.testing.assert_array_equal(
            result.data, [10, 20, 30]
        )

    def test_reverse_division(self):
        a = Tensor([1, 2, 3])

        result = 12 / a

        np.testing.assert_allclose(
            result.data, [12, 6, 4]
        )



if __name__ == "__main__":
    unittest.main()
