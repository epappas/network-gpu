# NetworkGPU - Remote GPU Device Driver

A high-performance virtual GPU device driver that enables PyTorch to utilize remote GPUs over the network as if they were local devices. This system allows for elastic GPU scaling, multi-cloud GPU access, and edge AI with remote compute resources.

## Features

- Seamless PyTorch Integration: Use remote GPUs with the same syntax as local CUDA devices
- High Performance: Built with Rust and async/await for maximum throughput
- Connection Pooling: Intelligent connection management with failover and load balancing  
- Memory Management: Efficient GPU memory pooling and automatic cleanup
- Secure Communication: TLS encryption and authentication support
- Monitoring & Stats: Built-in health checks and performance monitoring

## Getting Started - Step by Step

This guide will get you up and running with NetworkGPU. Follow each step carefully.

### Step 1: Prerequisites

Before starting, ensure you have the required dependencies:

#### Required Tools

```bash
# Check if Rust is installed
rustc --version
# If not installed: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Check if Python 3.8+ is installed
python3 --version

# Check if protobuf compiler is installed
protoc --version
```

#### Install Missing Prerequisites

**On Ubuntu/Debian:**

```bash
# Update package manager
sudo apt update

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install protobuf compiler
sudo apt install -y protobuf-compiler

# Install Python development headers
sudo apt install -y python3-dev python3-pip

# Install build essentials
sudo apt install -y build-essential pkg-config libssl-dev
```

**On macOS:**

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install protobuf compiler
brew install protobuf

# Install Python (if needed)
brew install python@3.11
```

**On Windows:**

```powershell
# Install Rust from https://rustup.rs/
# Install Python from https://python.org/downloads/
# Install protobuf compiler from https://grpc.io/docs/protoc-installation/
```

### Step 2: Clone and Setup Project

```bash
# Clone the repository
git clone <repository-url>
cd network-gpu

# Verify project structure
ls -la
# You should see: server/, client/, python-bindings/, proto/, examples/
```

### Step 3: Build Core Components

#### Option A: Quick Build (Recommended for Testing)

```bash
# Use the provided build script for automated setup
chmod +x build.sh
./build.sh
```

#### Option B: Manual Step-by-Step Build

If you prefer to build components individually:

```bash
# 1. Build protocol buffers
echo "Building protocol buffers..."
cd proto
cargo build
cd ..

# 2. Build client library
echo "Building client library..."
cd client  
cargo build
cd ..

# 3. Build server (requires CUDA for full functionality)
echo "Building server..."
cd server
cargo build
cd ..

# 4. Build Python bindings
echo "Building Python bindings..."
cd python-bindings
pip install maturin numpy
maturin develop --release
cd ..
```

### Step 4: Start the GPU Server

#### For Development/Testing (Mock GPU Mode)

```bash
# Start server with simulated GPU (no CUDA required)
cd server
cargo run --bin server -- --address 127.0.0.1:50051

# You should see output like:
# [INFO] Starting NetworkGPU Server on 127.0.0.1:50051
# [INFO] Initializing 1 GPU devices
# [INFO] Initialized GPU device 0: GPU Device 0
# [INFO] GPU service initialized successfully
```

#### For Production (Real GPU Mode)

```bash
# With CUDA toolkit installed
cd server
cargo build --release
./target/release/server --address 0.0.0.0:50051

# Check GPU detection
nvidia-smi  # Should show your GPUs
```

### Step 5: Verify Server is Running

Open a new terminal and test the server:

```bash
# Test server health
curl -v http://127.0.0.1:50051 2>&1 | grep "grpc"

# Or use grpcurl if available
grpcurl -plaintext 127.0.0.1:50051 list

# Check server logs for successful startup
# Look for: "GPU service initialized successfully"
```

### Step 6: Test Client Connection

#### Test Basic Compilation (No Server Required)

```bash
cd examples
python3 test_compilation.py

# Expected output:
# Successfully imported networkgpu
# Module imported, basic API available  
# All compilation tests passed!
```

#### Test Basic Client Operations

```bash
# Make sure server is running in another terminal
cd examples
python3 basic_usage.py

# Expected output:
# NetworkGPU Basic Usage Example
# Connected to NetworkGPU server
# Retrieved device info
# Basic operations completed successfully
```

### Step 7: PyTorch Integration (Advanced)

Once basic functionality works, test PyTorch integration:

```bash
# Install PyTorch if not already installed
pip install torch

# Test PyTorch integration
cd examples
python3 machine_learning.py

# Expected output:
# NetworkGPU PyTorch Integration Example  
# NetworkGPU backend registered
# Tensor operations on remote GPU successful
```

### Step 8: Verify Everything Works

Run the complete test suite:

```bash
# From project root
./test.sh

# Or run components individually:
cd server && cargo test
cd ../client && cargo test  
cd ../python-bindings && python -m pytest
```

## Usage Examples

### Basic Tensor Operations

```python
import networkgpu

# Connect to server (make sure it's running!)
client = networkgpu.Client(["grpc://127.0.0.1:50051"])

# Create tensors on remote GPU
x = client.create_tensor([1000, 1000], "float32", device_id=0)
y = client.create_tensor([1000, 1000], "float32", device_id=0)

# Perform operations
result = client.tensor_multiply(x, y)

# Get results back
data = client.get_tensor_data(result)
```

### PyTorch Integration

```python
import torch
import networkgpu

# Initialize NetworkGPU backend
networkgpu.init("grpc://127.0.0.1:50051")

# Use exactly like CUDA
x = torch.randn(1000, 1000, device="networkgpu:0")
y = torch.randn(1000, 1000, device="networkgpu:0")
z = x @ y  # Matrix multiplication on remote GPU
result = z.cpu()  # Transfer back to CPU
```

## Configuration

### Server Configuration

Create `server.toml` in the server directory:

```toml
[server]
address = "0.0.0.0:50051"
max_connections = 100
log_level = "info"

[gpu]
# Leave empty for auto-detection, or specify device IDs
device_ids = [0, 1, 2, 3]  
memory_pool_size_mb = 1024

[security]
enable_tls = false  # Set to true for production
```

### Client Configuration

```python
import networkgpu

# Single server
client = networkgpu.Client("grpc://gpu-server:50051")

# Multiple servers with load balancing
client = networkgpu.Client([
    "grpc://gpu-server-1:50051",
    "grpc://gpu-server-2:50051"
], max_connections=4)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Server Won't Start

```bash
# Check if port is in use
lsof -i :50051

# Kill existing process if needed
pkill -f "server.*50051"

# Check CUDA installation (for GPU mode)
nvidia-smi
nvcc --version
```

#### 2. Python Import Errors

```bash
# Reinstall Python bindings
cd python-bindings
pip uninstall networkgpu -y
maturin develop --release

# Check Python path
python3 -c "import sys; print(sys.path)"
```

#### 3. Connection Errors

```bash
# Test network connectivity
telnet 127.0.0.1 50051

# Check firewall settings
sudo ufw status  # Ubuntu
# Add rule if needed: sudo ufw allow 50051
```

#### 4. Build Errors

```bash
# Update Rust toolchain
rustup update stable

# Clean build cache
cargo clean

# Install missing dependencies
sudo apt install -y build-essential libssl-dev pkg-config
```

### Debug Mode

Enable debug logging for detailed information:

```bash
# Server debug mode
RUST_LOG=debug cargo run --bin server

# Client debug mode  
export RUST_LOG=debug
python3 your_script.py
```

## Performance Tuning

### Server Optimization

```toml
# server.toml
[performance]
worker_threads = 8
max_blocking_threads = 512
memory_pool_size_mb = 2048
enable_batch_processing = true
```

### Client Optimization

```python
# Use connection pooling
client = networkgpu.Client(
    servers=["grpc://server:50051"],
    max_connections=8,
    timeout_seconds=300,
    enable_compression=True
)
```

## Architecture Overview

```text
┌─────────────────┐    gRPC/TLS     ┌─────────────────┐
│   PyTorch App   │◄───────────────►│  GPU Server     │
│                 │                 │                 │
│  Python Client  │                 │  Rust Server    │
│  (Rust Binding) │                 │  GPU Manager    │
│                 │                 │                 │
│  Device Backend │                 │  CUDA Runtime   │
└─────────────────┘                 └─────────────────┘
```

### Component Details

- Server (`server/`): Rust-based GPU resource manager with CUDA integration
- Client (`client/`): High-performance Rust client library with connection pooling
- Python Bindings (`python-bindings/`): PyO3-based Python interface
- Protocol (`proto/`): gRPC service definitions and protocol buffers
- Examples (`examples/`): Usage examples and test scripts

## Next Steps

1. Production Deployment: Set up TLS, authentication, and monitoring
2. Scale Testing: Test with multiple GPUs and clients
3. Performance Benchmarking: Compare with local GPU performance
4. Integration: Integrate with your existing ML pipelines

## Getting Help

- Build Issues: Check `BUILD_AND_TEST.md` for detailed build instructions
- API Reference: See `TECHNICAL_IMPLEMENTATION_GUIDE.md` for API details
- Examples: Browse the `examples/` directory for usage patterns

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Rust](https://rust-lang.org/) and [Tokio](https://tokio.rs/)
- Uses [tonic](https://github.com/hyperium/tonic) for gRPC communication
- GPU operations via [cudarc](https://github.com/coreylowman/cudarc)
- Python bindings with [PyO3](https://pyo3.rs/)
- Inspired by PyTorch's CUDA device backend architecture
