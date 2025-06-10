#!/usr/bin/env python3
"""Basic tensor operations example using network-gpu."""

import network_gpu
import torch


def main():
    print("=== Basic Tensor Operations Example ===\n")

    # Create some example tensors
    print("1. Creating tensors...")
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)

    print(f"Tensor A:\n{a}")
    print(f"Tensor B:\n{b}")

    # Test tensor addition through Rust
    print("\n2. Adding tensors using Rust binding...")
    result = network_gpu.add_tensors(a, b)
    print(f"A + B =\n{result}")

    # Get tensor information
    print("\n3. Getting tensor information...")
    info_a = network_gpu.tensor_info(a)
    info_result = network_gpu.tensor_info(result)

    print(f"Tensor A info: {info_a}")
    print(f"Result info: {info_result}")

    # Test gradient computation
    print("\n4. Testing gradient computation...")
    loss = result.sum()
    loss.backward()

    print(f"Gradients for A:\n{a.grad}")
    print(f"Gradients for B:\n{b.grad}")

    print("\n✅ Basic tensor operations completed successfully!")


if __name__ == "__main__":
    main()
