#!/usr/bin/env python3
"""
Machine learning example using NetworkGPU.

This example demonstrates how to perform machine learning operations
on remote GPUs using NetworkGPU.
"""

import numpy as np
import networkgpu

def linear_layer_forward(x, weight, bias):
    """Simple linear layer forward pass."""
    return x @ weight.transpose(1, 0) + bias

def relu_activation(x):
    """ReLU activation function."""
    return x.relu()

def sigmoid_activation(x):
    """Sigmoid activation function."""
    return x.sigmoid()

def simple_neural_network():
    """Demonstrate a simple neural network on NetworkGPU."""
    print("Running simple neural network example...")
    
    client = networkgpu.init("grpc://localhost:50051")
    device_id = 0
    
    batch_size = 32
    input_size = 128
    hidden_size = 64
    output_size = 10
    
    print(f"Creating input data: batch_size={batch_size}, input_size={input_size}")
    x = networkgpu.randn(batch_size, input_size, device=device_id)
    
    print("Creating network weights...")
    w1 = networkgpu.randn(input_size, hidden_size, device=device_id) * 0.1
    b1 = networkgpu.zeros(hidden_size, device=device_id)
    
    w2 = networkgpu.randn(hidden_size, output_size, device=device_id) * 0.1
    b2 = networkgpu.zeros(output_size, device=device_id)
    
    print("Performing forward pass...")
    
    h1 = linear_layer_forward(x, w1.transpose(1, 0), b1)
    a1 = relu_activation(h1)
    print(f"Hidden layer output shape: {a1.shape}")
    
    h2 = linear_layer_forward(a1, w2.transpose(1, 0), b2)
    output = sigmoid_activation(h2)
    print(f"Network output shape: {output.shape}")
    
    output_cpu = output.cpu()
    print(f"Output range: {output_cpu.min():.4f} to {output_cpu.max():.4f}")
    
    networkgpu.synchronize(device_id)
    print("Neural network forward pass completed!")
    
    return output

def matrix_operations_benchmark():
    """Benchmark matrix operations on NetworkGPU."""
    print("\nRunning matrix operations benchmark...")
    
    client = networkgpu.init("grpc://localhost:50051")
    device_id = 0
    
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrices:")
        
        a = networkgpu.randn(size, size, device=device_id)
        b = networkgpu.randn(size, size, device=device_id)
        
        print("  Matrix multiplication...")
        c = a @ b
        
        print("  Element-wise addition...")
        d = a + b
        
        print("  Element-wise multiplication...")
        e = a * b
        
        print("  Sum reduction...")
        f = c.sum()
        
        print("  Sigmoid activation...")
        g = d.sigmoid()
        
        print("  ReLU activation...")
        h = e.relu()
        
        networkgpu.synchronize(device_id)
        print(f"  Completed {size}x{size} operations")

def convolutional_simulation():
    """Simulate convolutional operations using matrix operations."""
    print("\nSimulating convolutional operations...")
    
    client = networkgpu.init("grpc://localhost:50051")
    device_id = 0
    
    batch_size = 16
    height, width = 32, 32
    in_channels = 3
    out_channels = 64
    
    spatial_size = height * width
    
    print(f"Creating input data: {batch_size} images of {height}x{width}x{in_channels}")
    input_data = networkgpu.randn(batch_size, spatial_size * in_channels, device=device_id)
    
    conv_weights = networkgpu.randn(spatial_size * in_channels, spatial_size * out_channels, device=device_id) * 0.01
    
    print("Performing 'convolution' (matrix multiplication)...")
    conv_output = input_data @ conv_weights
    
    conv_output = conv_output.reshape([batch_size, spatial_size, out_channels])
    
    activated = conv_output.relu()
    
    pooled = activated.sum(dim=1) / spatial_size
    
    print(f"Final feature shape: {pooled.shape}")
    
    result_cpu = pooled.cpu()
    print(f"Feature range: {result_cpu.min():.4f} to {result_cpu.max():.4f}")
    
    networkgpu.synchronize(device_id)
    print("Convolutional simulation completed!")

def main():
    """Run all machine learning examples."""
    try:
        simple_neural_network()
        matrix_operations_benchmark()
        convolutional_simulation()
        
        print("\nAll machine learning examples completed successfully!")
        
    except Exception as e:
        print(f"Error in machine learning examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()