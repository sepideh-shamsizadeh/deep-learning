import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, dtype=np.float32):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):
        return (
            f"Tensor(data={self.data}, shape={self.shape}, "
            f"dtype={self.dtype}, requires_grad={self.requires_grad})"
        )



if __name__ == "__main__":
    x = Tensor([1, 2, 3])
    y = Tensor([1, 2, 3], dtype=np.float64)
    z = Tensor([1, 2, 3], requires_grad=True, dtype=np.float32)

    print(x.dtype)  # float32
    print(y.dtype)  # float64
    print(z.dtype)  # float32          
