#!/usr/bin/env python3
"""
Basic usage example of NetworkGPU.

This script demonstrates how to use NetworkGPU for remote GPU operations
that work similarly to local CUDA operations.
"""

import numpy as np
import networkgpu

def main():
    print("Initializing NetworkGPU...")
    client = networkgpu.init("grpc://localhost:50051")
    
    print("\nAvailable devices:")
    devices = networkgpu.list_devices()
    for device in devices:
        print(f"  {device}")
    
    if not devices:
        print("No devices available!")
        return
    
    device_id = 0
    
    print(f"\nUsing device {device_id}")
    
    print("Creating tensors...")
    x = networkgpu.randn(100, 100, device=device_id)
    y = networkgpu.randn(100, 100, device=device_id)
    
    print(f"Created tensors: x{x.shape}, y{y.shape}")
    
    print("Performing operations...")
    
    z1 = x + y
    print(f"Addition result: {z1.shape}")
    
    z2 = x @ y
    print(f"Matrix multiplication result: {z2.shape}")
    
    z3 = z1.sigmoid()
    print(f"Sigmoid result: {z3.shape}")
    
    z4 = z2.relu()
    print(f"ReLU result: {z4.shape}")
    
    print("Transferring results to CPU...")
    result_cpu = z2.cpu()
    print(f"CPU result shape: {result_cpu.shape}")
    print(f"CPU result type: {type(result_cpu)}")
    
    networkgpu.synchronize(device_id)
    print("Operations synchronized")
    
    print("Basic usage example completed successfully!")

def pytorch_example():
    """Example using PyTorch-like interface."""
    try:
        import torch
        import networkgpu
        
        print("\nPyTorch-style usage:")
        
        networkgpu.init("grpc://localhost:50051", register_pytorch_backend=True)
        
        x = torch.randn(50, 50, device="networkgpu:0")
        y = torch.randn(50, 50, device="networkgpu:0")
        
        z = x @ y
        result = z.cpu()
        
        print(f"PyTorch integration result: {result.shape}")
        
    except ImportError:
        print("PyTorch not available, skipping PyTorch example")
    except Exception as e:
        print(f"PyTorch example failed: {e}")

if __name__ == "__main__":
    main()
    pytorch_example()