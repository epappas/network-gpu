# Network-GPU Examples

This directory contains examples demonstrating the Python + Rust GPU neural network bindings.

## Examples

### 1. Basic Tensor Operations (`basic_tensor_ops.py`)
Demonstrates fundamental tensor operations:
- Creating PyTorch tensors
- Tensor addition through Rust bindings
- Tensor information extraction
- Gradient computation

```bash
uv run python examples/basic_tensor_ops.py
```

### 2. Linear Layer Demo (`linear_layer_demo.py`)
Shows how to use the Rust-implemented linear layer:
- Creating linear layers with Rust
- Forward and backward passes
- Parameter access and gradients
- Simple optimization step

```bash
uv run python examples/linear_layer_demo.py
```

### 3. Simple MLP (`simple_mlp.py`)
Complete training example with a multi-layer perceptron:
- Building models with multiple Rust layers
- Mini-batch training loop
- Loss computation and backpropagation
- Accuracy evaluation

```bash
uv run python examples/simple_mlp.py
```

## Running Examples

Make sure you have built the Rust extensions first:

```bash
uv sync
uv run maturin develop
```

Then run any example:

```bash
uv run python examples/<example_name>.py
```