---
title: "Pytorch"
layout: default
nav_order: 3
has_children: true
---

# Introduction to PyTorch

PyTorch is an open-source deep learning framework developed by Facebook's AI Research lab. It provides flexibility and speed when building deep learning models and is known for its easy-to-use API and dynamic computation graph.

## Why PyTorch?

- **Dynamic Computational Graph**: Build neural networks on-the-fly.
- **Pythonic**: Seamless integration with Python and popular libraries like NumPy.
- **Strong GPU Support**: Easy to move computations between CPU and GPU.
- **Extensive Community and Tools**: Access to tools like `torchvision`, `torchaudio`, and `torchtext`.

## Installing PyTorch

You can install PyTorch using pip:

```bash
pip install torch torchvision torchaudio
```

Or use the [official installation guide](https://pytorch.org/get-started/locally/).

## Tensors in PyTorch

Tensors are the basic building blocks in PyTorch. Think of them as NumPy arrays with additional capabilities for GPU acceleration.

```python
import torch

# Create a tensor
x = torch.tensor([1, 2, 3])
print(x)
```

## Basic Tensor Operations

```python
# Element-wise addition
a = torch.tensor([1.0, 2.0])
b = torch.tensor([3.0, 4.0])
c = a + b  # tensor([4., 6.])
```

## Neural Networks with `torch.nn`

PyTorch provides a module called `torch.nn` to help build and train neural networks easily.

```python
import torch.nn as nn

# Simple neural network
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)
```

## Training Loop (Simplified)

```python
import torch.optim as optim

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Resources

- [PyTorch Official Docs](https://pytorch.org/docs/)
- [Deep Learning with PyTorch Book](https://pytorch.org/deep-learning-with-pytorch)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

Happy Learning!
