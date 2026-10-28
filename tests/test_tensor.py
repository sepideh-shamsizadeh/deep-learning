
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


if __name__ == "__main__":
    unittest.main()
