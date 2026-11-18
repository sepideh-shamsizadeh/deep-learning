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

    def __add__(self, other):
        if isinstance(other, Tensor):
            other = other.data

        result = self.data + other
        return Tensor(result, dtype=result.dtype)


    def __sub__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        
        result = self.data - other
        return Tensor(result, dtype=result.dtype)
    
    def __mul__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        
        result = self.data * other
        return Tensor(result, dtype=result.dtype)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        
        result = self.data / other
        return Tensor(result, dtype=result.dtype)
    
    def __neg__(self):
        result = -self.data
        return Tensor(result, dtype=result.dtype)

    def __pow__(self, power):
        result = self.data ** power
        return Tensor(result, dtype=result.dtype)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        
        result = other - self.data
        return Tensor(result, dtype=result.dtype)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            other = other.data

        result = other / self.data
        return Tensor(result, dtype=result.dtype)
    
    def __matmul__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        
        result = self.data @ other
        return Tensor(result, dtype=result.dtype)

    def sum(self, axis=None, keepdims=False):
        result = np.sum(
            self.data,
            axis=axis,
            keepdims=keepdims
        )
        return Tensor(result, dtype=np.asarray(result).dtype)

    def mean(self, axis=None, keepdims=False):
        result = np.mean(
            self.data,
            axis=axis,
            keepdims=keepdims
        )
        return Tensor(result, dtype=np.asarray(result).dtype)



if __name__ == "__main__":
    x = Tensor([1, 2, 3])
    y = Tensor([1, 2, 3], dtype=np.float64)
    z = Tensor([1, 2, 3], requires_grad=True, dtype=np.float32)

    print(x.dtype)  # float32
    print(y.dtype)  # float64
    print(z.dtype)  # float32          
