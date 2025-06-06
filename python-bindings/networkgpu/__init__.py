"""
NetworkGPU - Remote GPU access for PyTorch

This package provides a PyTorch device backend that allows using remote GPUs
over the network as if they were local devices.

Example usage:
    import networkgpu
    
    # Initialize the backend
    networkgpu.init("grpc://gpu-server:50051")
    
    # Use remote GPUs like local ones
    x = torch.randn(1000, 1000, device="networkgpu:0")
    y = torch.randn(1000, 1000, device="networkgpu:0")
    z = x @ y  # Matrix multiplication on remote GPU
    result = z.cpu()  # Transfer back to CPU
"""

import os
import sys
from typing import List, Optional, Union

# Import the native module
from . import _networkgpu

# Re-export main classes
from ._networkgpu import (
    PyNetworkGPUClient as Client,
    PyNetworkGPUDevice as Device,
    PyRemoteTensor as Tensor,
    NetworkGPUError,
    DeviceError,
    TensorError,
    ConnectionError,
)

__version__ = "0.1.0"
__all__ = [
    "init",
    "list_devices", 
    "get_device",
    "tensor",
    "randn",
    "zeros",
    "ones",
    "Client",
    "Device", 
    "Tensor",
    "NetworkGPUError",
    "DeviceError",
    "TensorError",
    "ConnectionError",
]

# Global client instance
_global_client: Optional[Client] = None

def init(
    server_urls: Union[str, List[str]],
    max_connections: int = 4,
    timeout_seconds: int = 300,
    register_pytorch_backend: bool = True,
) -> Client:
    """
    Initialize NetworkGPU client and optionally register PyTorch backend.
    
    Args:
        server_urls: Server URL(s) to connect to
        max_connections: Maximum connections per server
        timeout_seconds: Request timeout in seconds  
        register_pytorch_backend: Whether to register as PyTorch backend
        
    Returns:
        NetworkGPU client instance
    """
    global _global_client
    
    if isinstance(server_urls, str):
        server_urls = [server_urls]
    
    _global_client = Client(server_urls, max_connections, timeout_seconds)
    
    if register_pytorch_backend:
        _register_pytorch_backend()
    
    return _global_client

def list_devices() -> List[Device]:
    """List available GPU devices."""
    if _global_client is None:
        raise RuntimeError("NetworkGPU not initialized. Call networkgpu.init() first.")
    return _global_client.list_devices()

def get_device(device_id: int) -> Device:
    """Get information about a specific device."""
    if _global_client is None:
        raise RuntimeError("NetworkGPU not initialized. Call networkgpu.init() first.")
    return _global_client.get_device_info(device_id)

def tensor(
    data,
    device: Union[str, int] = 0,
    dtype: str = "float32",
    requires_grad: bool = False,
) -> Tensor:
    """
    Create a tensor on a NetworkGPU device.
    
    Args:
        data: Input data (numpy array, list, or tensor)
        device: Device to place tensor on ("networkgpu:0" or 0)
        dtype: Data type
        requires_grad: Whether tensor requires gradients
        
    Returns:
        Remote tensor on NetworkGPU device
    """
    if _global_client is None:
        raise RuntimeError("NetworkGPU not initialized. Call networkgpu.init() first.")
    
    device_id = _parse_device(device)
    
    # Handle different input types
    import numpy as np
    if hasattr(data, 'numpy'):  # PyTorch tensor
        data = data.numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    
    return _global_client.tensor_from_numpy(data, device_id, requires_grad)

def randn(*shape, device: Union[str, int] = 0) -> Tensor:
    """Create a tensor filled with random normal values."""
    if _global_client is None:
        raise RuntimeError("NetworkGPU not initialized. Call networkgpu.init() first.")
    
    device_id = _parse_device(device)
    return _global_client.randn(list(shape), device_id)

def zeros(*shape, device: Union[str, int] = 0) -> Tensor:
    """Create a tensor filled with zeros."""
    if _global_client is None:
        raise RuntimeError("NetworkGPU not initialized. Call networkgpu.init() first.")
    
    device_id = _parse_device(device)
    return _global_client.zeros(list(shape), device_id)

def ones(*shape, device: Union[str, int] = 0) -> Tensor:
    """Create a tensor filled with ones."""
    if _global_client is None:
        raise RuntimeError("NetworkGPU not initialized. Call networkgpu.init() first.")
    
    device_id = _parse_device(device)
    return _global_client.ones(list(shape), device_id)

def synchronize(device: Union[str, int] = 0):
    """Synchronize all operations on a device."""
    if _global_client is None:
        raise RuntimeError("NetworkGPU not initialized. Call networkgpu.init() first.")
    
    device_id = _parse_device(device)
    _global_client.synchronize(device_id)

def _parse_device(device: Union[str, int]) -> int:
    """Parse device specification into device ID."""
    if isinstance(device, int):
        return device
    elif isinstance(device, str):
        if device.startswith("networkgpu:"):
            return int(device.split(":")[1])
        else:
            return int(device)
    else:
        raise ValueError(f"Invalid device specification: {device}")

def _register_pytorch_backend():
    """Register NetworkGPU as a PyTorch device backend."""
    try:
        import torch
        
        # Register the privateuse1 backend name
        torch._C._set_privateuse1_backend_name("networkgpu")
        
        # Register custom device module
        if hasattr(torch, '_register_device_module'):
            from . import pytorch_integration
            torch._register_device_module("networkgpu", pytorch_integration)
            
    except ImportError:
        print("Warning: PyTorch not found. PyTorch integration disabled.")
    except Exception as e:
        print(f"Warning: Failed to register PyTorch backend: {e}")

# Auto-initialize if environment variable is set
if "NETWORKGPU_SERVER" in os.environ:
    server_url = os.environ["NETWORKGPU_SERVER"]
    try:
        init(server_url)
        print(f"NetworkGPU auto-initialized with server: {server_url}")
    except Exception as e:
        print(f"Warning: Auto-initialization failed: {e}")