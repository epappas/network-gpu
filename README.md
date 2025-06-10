# Network-GPU

A high-performance Python + Rust library for GPU neural network operations with PyTorch integration.

## Features

- **Rust-powered performance**: Core tensor operations implemented in Rust for maximum speed
- **PyTorch integration**: Seamless interoperability with PyTorch tensors and gradients
- **Modular architecture**: Clean separation between tensor operations and neural network layers
- **GPU ready**: Built with GPU acceleration in mind
- **Modern tooling**: Latest versions of PyO3, maturin, and uv

## Quick Start

### Prerequisites

- **Python**: 3.11 or higher
- **Rust**: Latest stable version
- **uv**: Modern Python package manager

#### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd network-gpu
   ```

2. **Install dependencies and build**

   ```bash
   uv sync
   uv run maturin develop
   ```

3. **Verify installation**

   ```bash
   uv run python examples/basic_tensor_ops.py
   ```

## Project Structure

```text
network-gpu/
  crates/                     # Rust workspace crates
    tensor-ops/            # Core tensor operations
      src/lib.rs         # Add, multiply, matmul, sum, mean
      Cargo.toml
    neural-layers/         # Neural network layers
      src/lib.rs         # Linear, ReLU, Dropout layers
      Cargo.toml
  python/network_gpu/        # Python package
    __init__.py           # Python API exports
    examples/                  # Example scripts
      basic_tensor_ops.py   # Basic tensor operations
      linear_layer_demo.py  # Linear layer training
      simple_mlp.py         # Multi-layer perceptron
      advanced_layers.py    # All layer types
      README.md             # Examples documentation
    src/lib.rs                # Main PyO3 bindings
    pyproject.toml            # Python project configuration
    Cargo.toml                # Rust workspace configuration
  README.md                 # This file
```

## Development

### Setting Up Development Environment

1. **Clone and setup**

   ```bash
   git clone <repository-url>
   cd network-gpu
   uv sync
   ```

2. **Build in development mode**

   ```bash
   uv run maturin develop
   ```

3. **Run tests**

   ```bash
   uv run pytest
   ```

### Development Workflow

#### Making Changes

1. **Modify Rust code** in `crates/` directories
2. **Rebuild the extension**

   ```bash
   uv run maturin develop
   ```

3. **Test your changes**

   ```bash
   uv run python examples/basic_tensor_ops.py
   ```

#### Adding New Tensor Operations

1. **Add function to `crates/tensor-ops/src/lib.rs`**

   ```rust
   pub fn new_operation(py: Python, tensor: Bound<'_, PyAny>) -> PyResult<PyObject> {
       // Implementation here
       Ok(result.into_any().unbind())
   }
   ```

2. **Export in main `src/lib.rs`**

   ```rust
   #[pyfunction]
   fn new_operation(py: Python, tensor: Bound<'_, PyAny>) -> PyResult<PyObject> {
       tensor_ops::new_operation(py, tensor)
   }
   
   // Add to module
   m.add_function(wrap_pyfunction!(new_operation, m)?)?;
   ```

3. **Update Python exports in `python/network_gpu/__init__.py`**

   ```python
   from ._network_gpu import new_operation
   __all__.append("new_operation")
   ```

#### Adding New Neural Network Layers

1. **Add layer to `crates/neural-layers/src/lib.rs`**

   ```rust
   #[pyclass]
   pub struct NewLayer {
       // Layer parameters
   }
   
   #[pymethods]
   impl NewLayer {
       #[new]
       pub fn new(/* parameters */) -> Self { /* */ }
       
       pub fn forward(&self, py: Python, input: Bound<'_, PyAny>) -> PyResult<PyObject> {
           // Forward pass implementation
       }
   }
   ```

2. **Export layer and creator function**
3. **Update main bindings and Python exports**

### Testing

#### Run Examples

```bash
# Basic operations
uv run python examples/basic_tensor_ops.py

# Layer demonstrations
uv run python examples/linear_layer_demo.py
uv run python examples/advanced_layers.py

# Complete training example
uv run python examples/simple_mlp.py
```

#### Performance Testing

```bash
# Time operations
uv run python -c "
import time
import torch
import network_gpu

# Test Rust vs PyTorch performance
a = torch.randn(1000, 1000)
b = torch.randn(1000, 1000)

# Rust implementation
start = time.time()
result_rust = network_gpu.add_tensors(a, b)
rust_time = time.time() - start

# PyTorch implementation  
start = time.time()
result_torch = torch.add(a, b)
torch_time = time.time() - start

print(f'Rust: {rust_time:.6f}s, PyTorch: {torch_time:.6f}s')
"
```

### Building for Production

#### Release Build

```bash
uv run maturin build --release
```

#### Build Python Wheel

```bash
uv run maturin build --release --out dist/
pip install dist/*.whl
```

## API Reference

### Tensor Operations

```python
import network_gpu

# Basic operations
result = network_gpu.add_tensors(a, b)
result = network_gpu.multiply_tensors(a, b)
result = network_gpu.matmul_tensors(a, b)

# Reductions
sum_result = network_gpu.tensor_sum(tensor, dim=1)
mean_result = network_gpu.tensor_mean(tensor, dim=0)

# Information
info = network_gpu.tensor_info(tensor)
```

### Neural Network Layers

```python
# Linear layer
layer = network_gpu.create_linear_layer(input_dim=10, output_dim=5, bias=True)
output = layer.forward(input_tensor)
params = layer.parameters()

# Activation layers
relu = network_gpu.create_relu_layer()
output = relu.forward(input_tensor)

# Regularization
dropout = network_gpu.create_dropout_layer(p=0.5)
output = dropout.forward(input_tensor)
dropout.eval()  # Switch to evaluation mode
```

## Dependencies

### Python Dependencies

- `torch>=2.5.0` - PyTorch for tensor operations
- `numpy>=2.3.0` - Numerical computing

### Rust Dependencies

- `pyo3>=0.25` - Python-Rust bindings
- `numpy>=0.25` - NumPy integration

### Development Dependencies

- `pytest>=8.4.0` - Testing framework
- `maturin>=1.7.0` - Rust-Python build tool

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the development workflow above
4. Add tests for new functionality
5. Run the test suite (`uv run pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- **Rust**: Follow standard Rust conventions (`rustfmt`)
- **Python**: Follow PEP 8 (enforced by `ruff`)
- **Documentation**: Include docstrings for all public functions

## Troubleshooting

### Build Issues

**Error: `maturin develop` fails**

```bash
# Clean and rebuild
rm -rf target/
rm Cargo.lock
uv run maturin develop
```

**Error: PyO3 version conflicts**

```bash
# Update all dependencies
uv sync --upgrade
```

### Runtime Issues

**Error: Module import fails**

```bash
# Ensure you've built the extension
uv run maturin develop

# Check Python path
uv run python -c "import network_gpu; print(network_gpu.__file__)"
```

**Error: CUDA/GPU not available**

- This library uses PyTorch for GPU operations
- Ensure PyTorch is installed with CUDA support
- Check `torch.cuda.is_available()`

## Performance Notes

- Rust operations have lower overhead for small tensors
- For large tensors, PyTorch's optimized kernels may be faster
- Use this library when you need custom operations or want to leverage Rust's safety
- GPU acceleration happens through PyTorch's CUDA integration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v0.1.0

- Initial release
- Basic tensor operations (add, multiply, matmul, sum, mean)
- Neural network layers (Linear, ReLU, Dropout)
- PyTorch gradient integration
- Modular Rust workspace architecture
