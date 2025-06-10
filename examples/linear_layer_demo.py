#!/usr/bin/env python3
"""Linear layer demonstration using network-gpu."""

import network_gpu
import torch
import torch.nn.functional as F


def main():
    print("=== Linear Layer Demo ===\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create a simple dataset
    print("1. Creating synthetic dataset...")
    batch_size, input_dim, output_dim = 32, 4, 2
    X = torch.randn(batch_size, input_dim, requires_grad=True)
    y = torch.randint(0, output_dim, (batch_size,))

    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Create linear layer using Rust
    print("\n2. Creating linear layer with Rust...")
    layer = network_gpu.create_linear_layer(input_dim, output_dim, bias=True)

    # Get layer parameters
    weight = layer.weight
    bias = layer.bias
    params = layer.parameters()

    print(f"Weight shape: {weight.shape}")
    print(f"Bias shape: {bias.shape}")
    print(f"Total parameters: {len(params)}")

    # Forward pass
    print("\n3. Forward pass...")
    logits = layer.forward(X)
    print(f"Output shape: {logits.shape}")
    print(f"Output sample:\n{logits[:5]}")

    # Compute loss
    print("\n4. Computing loss...")
    loss = F.cross_entropy(logits, y)
    print(f"Cross-entropy loss: {loss.item():.4f}")

    # Backward pass
    print("\n5. Backward pass...")
    loss.backward()

    # Check gradients
    print(f"Weight gradients shape: {weight.grad.shape}")
    print(f"Bias gradients shape: {bias.grad.shape}")
    print(f"Weight gradient norm: {weight.grad.norm().item():.4f}")
    print(f"Bias gradient norm: {bias.grad.norm().item():.4f}")

    # Simple training step
    print("\n6. Simple optimization step...")
    learning_rate = 0.01
    with torch.no_grad():
        weight -= learning_rate * weight.grad
        bias -= learning_rate * bias.grad

        # Zero gradients
        weight.grad.zero_()
        bias.grad.zero_()

    # Test after update
    new_logits = layer.forward(X)
    new_loss = F.cross_entropy(new_logits, y)
    print(f"Loss after one step: {new_loss.item():.4f}")
    print(f"Loss change: {(new_loss - loss).item():.4f}")

    print("\n✅ Linear layer demo completed successfully!")


if __name__ == "__main__":
    main()
