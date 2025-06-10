#!/usr/bin/env python3
"""Simple MLP example using network-gpu layers."""

import network_gpu
import torch
import torch.nn.functional as F


class SimpleMLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.layer1 = network_gpu.create_linear_layer(input_dim, hidden_dim, bias=True)
        self.layer2 = network_gpu.create_linear_layer(hidden_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1.forward(x)
        x = F.relu(x)
        x = self.layer2.forward(x)
        return x

    def parameters(self):
        params = []
        params.extend(self.layer1.parameters())
        params.extend(self.layer2.parameters())
        return params


def main():
    print("=== Simple MLP Training Example ===\n")

    torch.manual_seed(42)

    # Create synthetic classification dataset
    print("1. Creating dataset...")
    n_samples, input_dim, hidden_dim, output_dim = 1000, 10, 64, 3

    # Generate random data with some structure
    X = torch.randn(n_samples, input_dim)
    # Create targets based on simple rules
    y = ((X[:, 0] + X[:, 1] > 0).long() + (X[:, 2] + X[:, 3] > 0).long()).clamp(0, output_dim - 1)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {output_dim} classes")
    print(f"Class distribution: {torch.bincount(y)}")

    # Create model
    print("\n2. Creating MLP model...")
    model = SimpleMLP(input_dim, hidden_dim, output_dim)
    params = model.parameters()
    print(f"Model parameters: {len(params)} tensors")

    # Training setup
    learning_rate = 0.01
    epochs = 100
    batch_size = 32

    print(f"\n3. Training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        # Mini-batch training
        for i in range(0, len(X), batch_size):
            batch_X = X[i : i + batch_size]
            batch_y = y[i : i + batch_size]

            if len(batch_X) == 0:
                continue

            # Forward pass
            logits = model.forward(batch_X)
            loss = F.cross_entropy(logits, batch_y)

            # Backward pass
            loss.backward()

            # Update parameters
            with torch.no_grad():
                for param in params:
                    if param.grad is not None:
                        param -= learning_rate * param.grad
                        param.grad.zero_()

            total_loss += loss.item()
            n_batches += 1

        # Print progress
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / n_batches

            # Compute accuracy
            with torch.no_grad():
                logits = model.forward(X)
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == y).float().mean().item()

            print(f"Epoch {epoch + 1:3d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

    # Final evaluation
    print("\n4. Final evaluation...")
    with torch.no_grad():
        logits = model.forward(X)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean().item()

        # Per-class accuracy
        for class_id in range(output_dim):
            mask = y == class_id
            if mask.sum() > 0:
                class_acc = (predictions[mask] == y[mask]).float().mean().item()
                print(f"Class {class_id} accuracy: {class_acc:.4f}")

    print(f"\nOverall accuracy: {accuracy:.4f}")
    print("\n✅ MLP training completed successfully!")


if __name__ == "__main__":
    main()
