"""
PyTorch integration for NetworkGPU.

This module provides PyTorch device backend integration, allowing PyTorch
to use NetworkGPU devices as if they were local GPUs.
"""

import torch
from typing import Any, Dict, Optional, Union
import networkgpu

class NetworkGPUDevice:
    """NetworkGPU device representation for PyTorch."""
    
    def __init__(self, device_id: int):
        self.device_id = device_id
        self.type = "networkgpu"
    
    def __str__(self):
        return f"networkgpu:{self.device_id}"
    
    def __repr__(self):
        return f"NetworkGPUDevice(device_id={self.device_id})"
    
    def __eq__(self, other):
        return isinstance(other, NetworkGPUDevice) and self.device_id == other.device_id
    
    def __hash__(self):
        return hash((self.type, self.device_id))

def device_count() -> int:
    """Get the number of available NetworkGPU devices."""
    try:
        devices = networkgpu.list_devices()
        return len(devices)
    except:
        return 0

def get_device_name(device_id: int) -> str:
    """Get the name of a NetworkGPU device."""
    try:
        device = networkgpu.get_device(device_id)
        return device.name
    except:
        return f"NetworkGPU Device {device_id}"

def get_device_capability(device_id: int) -> tuple:
    """Get the compute capability of a NetworkGPU device."""
    try:
        device = networkgpu.get_device(device_id)
        return (device.compute_capability_major, device.compute_capability_minor)
    except:
        return (0, 0)

def synchronize(device: Optional[Union[int, NetworkGPUDevice]] = None):
    """Synchronize operations on a NetworkGPU device."""
    if device is None:
        device_id = torch.cuda.current_device() if hasattr(torch.cuda, 'current_device') else 0
    elif isinstance(device, NetworkGPUDevice):
        device_id = device.device_id
    else:
        device_id = int(device)
    
    networkgpu.synchronize(device_id)

def empty_cache():
    """Clear the memory cache (NetworkGPU specific)."""
    if networkgpu._global_client:
        networkgpu._global_client.cleanup_cache()

def memory_stats(device: Optional[Union[int, NetworkGPUDevice]] = None) -> Dict[str, Any]:
    """Get memory statistics for a NetworkGPU device."""
    if device is None:
        device_id = 0
    elif isinstance(device, NetworkGPUDevice):
        device_id = device.device_id
    else:
        device_id = int(device)
    
    try:
        device_info = networkgpu.get_device(device_id)
        return {
            "allocated_bytes.all.current": device_info.total_memory - device_info.free_memory,
            "allocated_bytes.all.peak": device_info.total_memory - device_info.free_memory,
            "reserved_bytes.all.current": device_info.total_memory,
            "reserved_bytes.all.peak": device_info.total_memory,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        }
    except:
        return {}

def get_device_properties(device: Union[int, NetworkGPUDevice]) -> object:
    """Get device properties."""
    if isinstance(device, NetworkGPUDevice):
        device_id = device.device_id
    else:
        device_id = int(device)
    
    try:
        device_info = networkgpu.get_device(device_id)
        
        class DeviceProperties:
            def __init__(self, device_info):
                self.name = device_info.name
                self.major = device_info.compute_capability_major
                self.minor = device_info.compute_capability_minor
                self.total_memory = device_info.total_memory
                self.multi_processor_count = device_info.multiprocessor_count
                self.max_threads_per_block = device_info.max_threads_per_block
                self.warp_size = device_info.warp_size
        
        return DeviceProperties(device_info)
    except:
        return None

# Custom tensor creation functions
def _tensor_constructor_networkgpu(
    data,
    dtype=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    """Create a tensor on NetworkGPU device."""
    if device is None:
        device_id = 0
    elif isinstance(device, str):
        if device.startswith("networkgpu:"):
            device_id = int(device.split(":")[1])
        else:
            device_id = int(device)
    elif isinstance(device, NetworkGPUDevice):
        device_id = device.device_id
    else:
        device_id = int(device)
    
    # Convert data to numpy if needed
    import numpy as np
    if hasattr(data, 'numpy'):
        data = data.numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Create NetworkGPU tensor
    tensor = networkgpu.tensor(data, device_id, requires_grad=requires_grad)
    
    # Wrap in PyTorch tensor-like interface
    return NetworkGPUTensorWrapper(tensor)

class NetworkGPUTensorWrapper:
    """Wrapper to make NetworkGPU tensors compatible with PyTorch."""
    
    def __init__(self, networkgpu_tensor):
        self._tensor = networkgpu_tensor
        self.device = NetworkGPUDevice(networkgpu_tensor.device_id)
        self.dtype = self._map_dtype(networkgpu_tensor.dtype)
        self.requires_grad = networkgpu_tensor.requires_grad
    
    def _map_dtype(self, dtype_str):
        """Map NetworkGPU dtype to PyTorch dtype."""
        mapping = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
            "uint8": torch.uint8,
        }
        return mapping.get(dtype_str, torch.float32)
    
    @property
    def shape(self):
        return torch.Size(self._tensor.shape)
    
    @property
    def size(self):
        return self.shape
    
    def dim(self):
        return len(self._tensor.shape)
    
    def numel(self):
        return int(torch.prod(torch.tensor(self._tensor.shape)))
    
    def cpu(self):
        """Transfer tensor to CPU."""
        import torch
        import numpy as np
        
        # Get data from NetworkGPU
        cpu_data = self._tensor.cpu()
        
        # Convert to PyTorch tensor
        if isinstance(cpu_data, np.ndarray):
            return torch.from_numpy(cpu_data)
        else:
            # Handle raw bytes
            return torch.frombuffer(cpu_data, dtype=self.dtype).view(self.shape)
    
    def cuda(self, device=None):
        """Transfer tensor to CUDA device (if available)."""
        return self.cpu().cuda(device)
    
    def to(self, device=None, dtype=None, **kwargs):
        """Transfer tensor to device/dtype."""
        if device is not None and str(device).startswith("networkgpu"):
            return self  # Already on NetworkGPU
        else:
            cpu_tensor = self.cpu()
            if device is not None:
                cpu_tensor = cpu_tensor.to(device)
            if dtype is not None:
                cpu_tensor = cpu_tensor.to(dtype)
            return cpu_tensor
    
    # Arithmetic operations
    def __add__(self, other):
        if isinstance(other, NetworkGPUTensorWrapper):
            result = self._tensor.add(other._tensor)
            return NetworkGPUTensorWrapper(result)
        else:
            return self.cpu() + other
    
    def __sub__(self, other):
        if isinstance(other, NetworkGPUTensorWrapper):
            result = self._tensor.sub(other._tensor)
            return NetworkGPUTensorWrapper(result)
        else:
            return self.cpu() - other
    
    def __mul__(self, other):
        if isinstance(other, NetworkGPUTensorWrapper):
            result = self._tensor.mul(other._tensor)
            return NetworkGPUTensorWrapper(result)
        else:
            return self.cpu() * other
    
    def __truediv__(self, other):
        if isinstance(other, NetworkGPUTensorWrapper):
            result = self._tensor.div(other._tensor)
            return NetworkGPUTensorWrapper(result)
        else:
            return self.cpu() / other
    
    def __matmul__(self, other):
        if isinstance(other, NetworkGPUTensorWrapper):
            result = self._tensor.matmul(other._tensor)
            return NetworkGPUTensorWrapper(result)
        else:
            return self.cpu() @ other
    
    def transpose(self, dim0, dim1):
        result = self._tensor.transpose(dim0, dim1)
        return NetworkGPUTensorWrapper(result)
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        result = self._tensor.reshape(list(shape))
        return NetworkGPUTensorWrapper(result)
    
    def sum(self, dim=None, keepdim=False):
        result = self._tensor.sum(dim)
        return NetworkGPUTensorWrapper(result)
    
    def sigmoid(self):
        result = self._tensor.sigmoid()
        return NetworkGPUTensorWrapper(result)
    
    def relu(self):
        result = self._tensor.relu()
        return NetworkGPUTensorWrapper(result)
    
    def __str__(self):
        return f"tensor({self._tensor.shape}, device='{self.device}', dtype={self.dtype})"
    
    def __repr__(self):
        return self.__str__()

# Register hooks for PyTorch integration
try:
    import torch
    
    # Override tensor creation on networkgpu device
    original_tensor = torch.tensor
    
    def networkgpu_tensor(*args, device=None, **kwargs):
        if device is not None and str(device).startswith("networkgpu"):
            return _tensor_constructor_networkgpu(*args, device=device, **kwargs)
        else:
            return original_tensor(*args, device=device, **kwargs)
    
    # Monkey patch for demonstration (in production, this would use proper PyTorch extension mechanisms)
    torch.tensor = networkgpu_tensor
    
except ImportError:
    pass  # PyTorch not available