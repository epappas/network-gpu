#!/usr/bin/env python3
"""Advanced neural network layers demonstration."""

import network_gpu
import torch


def main():
    print("=== Advanced Layers Demo ===\n")

    torch.manual_seed(42)

    # Create test data
    print("1. Creating test data...")
    batch_size, input_dim = 16, 10
    X = torch.randn(batch_size, input_dim, requires_grad=True)
    print(f"Input shape: {X.shape}")

    # Test additional tensor operations
    print("\n2. Testing tensor operations...")

    # Element-wise multiplication
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

    mult_result = network_gpu.multiply_tensors(a, b)
    print(f"Element-wise multiply:\n{mult_result}")

    # Matrix multiplication
    matmul_result = network_gpu.matmul_tensors(a, b.T)
    print(f"Matrix multiply:\n{matmul_result}")

    # Tensor sum and mean
    sum_result = network_gpu.tensor_sum(X, dim=1)
    mean_result = network_gpu.tensor_mean(X, dim=0)
    print(f"Sum along dim 1 shape: {sum_result.shape}")
    print(f"Mean along dim 0 shape: {mean_result.shape}")

    # Test ReLU layer
    print("\n3. Testing ReLU layer...")
    relu = network_gpu.create_relu_layer()

    test_input = torch.tensor([[-1.0, 0.0, 1.0, 2.0]])
    relu_output = relu.forward(test_input)
    print(f"ReLU input: {test_input}")
    print(f"ReLU output: {relu_output}")

    # Test Dropout layer
    print("\n4. Testing Dropout layer...")
    dropout = network_gpu.create_dropout_layer(0.5)

    print(f"Dropout training mode: {dropout.training}")

    # Training mode (dropout active)
    dropout_train_output = dropout.forward(X)
    print(f"Dropout (training) output sample:\n{dropout_train_output[:3, :5]}")

    # Evaluation mode (dropout inactive)
    dropout.eval()
    print(f"Dropout training mode after eval(): {dropout.training}")
    dropout_eval_output = dropout.forward(X)
    print(f"Dropout (eval) output sample:\n{dropout_eval_output[:3, :5]}")

    # Build a complete network
    print("\n5. Building complete network with all layers...")

    class AdvancedNet:
        def __init__(self):
            self.layer1 = network_gpu.create_linear_layer(input_dim, 32, bias=True)
            self.relu1 = network_gpu.create_relu_layer()
            self.dropout1 = network_gpu.create_dropout_layer(0.3)
            self.layer2 = network_gpu.create_linear_layer(32, 16, bias=True)
            self.relu2 = network_gpu.create_relu_layer()
            self.dropout2 = network_gpu.create_dropout_layer(0.2)
            self.layer3 = network_gpu.create_linear_layer(16, 3, bias=True)

        def forward(self, x):
            x = self.layer1.forward(x)
            x = self.relu1.forward(x)
            x = self.dropout1.forward(x)
            x = self.layer2.forward(x)
            x = self.relu2.forward(x)
            x = self.dropout2.forward(x)
            x = self.layer3.forward(x)
            return x

        def eval_mode(self):
            self.dropout1.eval()
            self.dropout2.eval()

        def train_mode(self):
            self.dropout1.train()
            self.dropout2.train()

    net = AdvancedNet()

    # Forward pass in training mode
    print("Forward pass (training mode)...")
    output_train = net.forward(X)
    print(f"Output shape: {output_train.shape}")
    print(f"Output sample:\n{output_train[:3]}")

    # Forward pass in eval mode
    print("\nForward pass (eval mode)...")
    net.eval_mode()
    output_eval = net.forward(X)
    print(f"Output sample:\n{output_eval[:3]}")

    # Test gradient flow
    print("\n6. Testing gradient flow...")
    net.train_mode()
    output = net.forward(X)
    loss = output.sum()
    loss.backward()

    # Check gradients in first layer
    weight1 = net.layer1.weight
    print(f"Layer 1 weight gradient norm: {weight1.grad.norm().item():.6f}")

    print("\n✅ Advanced layers demo completed successfully!")


if __name__ == "__main__":
    main()
