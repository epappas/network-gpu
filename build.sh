#!/bin/bash
# Complete build script for NetworkGPU

set -e

echo "Building NetworkGPU System..."

echo "Checking prerequisites..."
command -v rustc >/dev/null 2>&1 || { echo "Rust not found. Install from https://rustup.rs/"; exit 1; }
command -v protoc >/dev/null 2>&1 || { echo "protoc not found. Install protobuf compiler"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python 3 not found"; exit 1; }

echo "Prerequisites check passed"

echo "Adding missing dependencies..."

if ! grep -q "toml.*=" server/Cargo.toml; then
    cat >> server/Cargo.toml << 'EOF'

# Additional dependencies
toml = "0.8"
serde_json = "1.0"
serde_yaml = "0.9"
fastrand = "2.0"
tokio-stream = "0.1"
EOF
fi

if ! grep -q "bytes.*=" client/Cargo.toml; then
    cat >> client/Cargo.toml << 'EOF'

# Additional client dependencies  
bytes = "1.0"
fastrand = "2.0"
EOF
fi

echo "Dependencies added"

echo "Building protocol buffers..."
cd proto
cargo build --release
cd ..
echo "Protocol buffers built"

echo "Building server..."
cd server
cargo build --release
if [ $? -eq 0 ]; then
    echo "Server built successfully"
else
    echo "Server build failed - continuing with client"
fi
cd ..

echo "Building client..."
cd client
cargo build --release
cd ..
echo "Client built"

echo "Building Python bindings..."
cd python-bindings

if command -v maturin >/dev/null 2>&1; then
    echo "Using maturin for Python bindings..."
    maturin develop --release
    if [ $? -eq 0 ]; then
        echo "Python bindings built successfully"
    else
        echo "Python bindings build failed - skipping"
    fi
else
    echo "maturin not found - installing..."
    pip install maturin
    if [ $? -eq 0 ]; then
        maturin develop --release
        echo "Python bindings built"
    else
        echo "Failed to install maturin - skipping Python bindings"
    fi
fi

cd ..

echo "Running basic tests..."
cd proto && cargo test --release && cd ..
cd client && cargo test --release && cd ..

if [ -f "server/target/release/server" ] || [ -f "target/release/server" ]; then
    echo "Build completed successfully!"
    echo ""
    echo "To run the server:"
    echo "  cd server && cargo run --bin server"
    echo "  or"
    echo "  ./target/release/server"
    echo ""
    echo "To test the Python bindings:"
    echo "  cd examples && python3 test_compilation.py"
else
    echo "Build completed with some failures"
    echo "Check individual component builds"
fi