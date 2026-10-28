# Deep Learning from Scratch

Building a lightweight deep learning framework from scratch in Python and NumPy to explore the mathematical foundations, internal mechanisms, and engineering principles behind modern deep learning libraries.

## Overview

This repository documents my journey from implementing individual deep learning algorithms to building **MiniDL**, a small, modular deep learning framework.

The goal is to understand not only how to train neural networks, but also how deep learning frameworks work internally, including tensor operations, automatic differentiation, backpropagation, and optimization.

## Repository Structure

```text
deep-learning/
├── legacy/            # Previous deep learning implementations
├── minidl/            # MiniDL framework
│   └── __init__.py
├── tests/             # Unit tests and gradient verification
├── README.md
└── pyproject.toml
```

## MiniDL — Deep Learning Framework

MiniDL is an educational deep learning framework built using Python and NumPy.

### Planned Features

- **Tensor Engine:** Multidimensional arrays and mathematical operations.
- **Automatic Differentiation:** Dynamic computation graphs and reverse-mode autodiff.
- **Neural Network Layers:** Linear layers and activation functions.
- **Loss Functions:** Mean Squared Error and Cross-Entropy.
- **Optimizers:** SGD, Momentum, and Adam.
- **Training:** End-to-end neural network training.
- **Testing:** Unit tests and numerical gradient checking.
- **Performance:** Profiling and comparisons with PyTorch.

### Development Roadmap

- [ ] Initialize the MiniDL package
- [ ] Implement Tensor class
- [ ] Implement tensor operations
- [ ] Build computation graph
- [ ] Implement automatic differentiation
- [ ] Implement activation functions
- [ ] Implement neural network layers
- [ ] Implement loss functions
- [ ] Implement optimizers
- [ ] Train a neural network
- [ ] Validate gradients and benchmark performance

## Previous Implementations

The `legacy/` directory preserves earlier experiments implementing neural network components and algorithms from scratch, including:

- Activation functions and derivatives
- Forward and backward propagation
- Gradient-based optimization
- Convolution operations
- Basic neural network architectures

These implementations provide a foundation for developing MiniDL.

## Technologies

- **Python** — Framework implementation
- **NumPy** — Numerical computation and array operations
- **pytest** — Unit testing
- **PyTorch** — Reference framework for validation and benchmarking

## Learning Objectives

This project focuses on developing a deeper understanding of:

1. Tensor operations, shapes, broadcasting, and memory.
2. Computational graphs and the chain rule.
3. Reverse-mode automatic differentiation.
4. Neural network architecture and optimization.
5. Numerical correctness and gradient verification.
6. Performance considerations in deep learning systems.

## Development Workflow

Development is organized using GitHub Issues and feature branches.

Each major component follows this process:

**Issue → Feature Branch → Implementation → Testing → Pull Request → Merge**

## References

- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [micrograd — Andrej Karpathy](https://github.com/karpathy/micrograd)

## Project Status

**Under active development.**

MiniDL is intended for educational purposes and is not a replacement for production deep learning frameworks.