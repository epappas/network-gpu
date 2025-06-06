use networkgpu_client::NetworkGPUClient;
use networkgpu_proto::DeviceInfo;
use pyo3::prelude::*;
use std::sync::Arc;

use crate::error::PyClientError;
use crate::runtime::RUNTIME;

#[pyclass]
#[derive(Clone)]
pub struct PyNetworkGPUDevice {
    device_info: DeviceInfo,
    client: Arc<NetworkGPUClient>,
}

#[pymethods]
impl PyNetworkGPUDevice {
    /// Device ID
    #[getter]
    fn device_id(&self) -> i32 {
        self.device_info.device_id
    }
    
    /// Device name
    #[getter]
    fn name(&self) -> String {
        self.device_info.name.clone()
    }
    
    /// Total memory in bytes
    #[getter]
    fn total_memory(&self) -> u64 {
        self.device_info.total_memory
    }
    
    /// Free memory in bytes
    #[getter]
    fn free_memory(&self) -> u64 {
        self.device_info.free_memory
    }
    
    /// Compute capability major version
    #[getter]
    fn compute_capability_major(&self) -> i32 {
        self.device_info.compute_capability_major
    }
    
    /// Compute capability minor version
    #[getter]
    fn compute_capability_minor(&self) -> i32 {
        self.device_info.compute_capability_minor
    }
    
    /// Number of multiprocessors
    #[getter]
    fn multiprocessor_count(&self) -> i32 {
        self.device_info.multiprocessor_count
    }
    
    /// Maximum threads per block
    #[getter]
    fn max_threads_per_block(&self) -> i32 {
        self.device_info.max_threads_per_block
    }
    
    /// Warp size
    #[getter]
    fn warp_size(&self) -> i32 {
        self.device_info.warp_size
    }
    
    /// Whether concurrent kernels are supported
    #[getter]
    fn concurrent_kernels(&self) -> bool {
        self.device_info.concurrent_kernels
    }
    
    /// Whether ECC is enabled
    #[getter]
    fn ecc_enabled(&self) -> bool {
        self.device_info.ecc_enabled
    }
    
    /// Device UUID
    #[getter]
    fn uuid(&self) -> String {
        self.device_info.uuid.clone()
    }
    
    /// Allocate this device for exclusive use
    fn allocate(&self, client_id: String, memory_limit: Option<u64>) -> PyResult<String> {
        let token = RUNTIME.block_on(async {
            self.client.allocate_device(
                self.device_info.device_id,
                client_id,
                memory_limit.unwrap_or(self.device_info.total_memory),
            ).await
        }).map_err(PyClientError::from)?;
        
        Ok(token)
    }
    
    /// Release device allocation
    fn release(&self) -> PyResult<()> {
        RUNTIME.block_on(async {
            self.client.release_device(self.device_info.device_id).await
        }).map_err(PyClientError::from)?;
        
        Ok(())
    }
    
    /// Synchronize all operations on this device
    fn synchronize(&self) -> PyResult<()> {
        RUNTIME.block_on(async {
            self.client.synchronize_device(self.device_info.device_id).await
        }).map_err(PyClientError::from)?;
        
        Ok(())
    }
    
    /// Refresh device information
    fn refresh(&mut self) -> PyResult<()> {
        let device_info = RUNTIME.block_on(async {
            self.client.get_device_info(self.device_info.device_id).await
        }).map_err(PyClientError::from)?;
        
        self.device_info = device_info;
        Ok(())
    }
    
    /// Get device memory utilization as a percentage
    fn memory_utilization(&self) -> f64 {
        if self.device_info.total_memory == 0 {
            return 0.0;
        }
        
        let used_memory = self.device_info.total_memory - self.device_info.free_memory;
        (used_memory as f64 / self.device_info.total_memory as f64) * 100.0
    }
    
    /// Check if device has sufficient free memory
    fn has_free_memory(&self, required_bytes: u64) -> bool {
        self.device_info.free_memory >= required_bytes
    }
    
    /// Get device properties as a dictionary
    fn properties(&self) -> PyResult<std::collections::HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            let mut props = std::collections::HashMap::new();
            
            props.insert("device_id".to_string(), self.device_info.device_id.into_py(py));
            props.insert("name".to_string(), self.device_info.name.clone().into_py(py));
            props.insert("total_memory".to_string(), self.device_info.total_memory.into_py(py));
            props.insert("free_memory".to_string(), self.device_info.free_memory.into_py(py));
            props.insert("compute_capability_major".to_string(), self.device_info.compute_capability_major.into_py(py));
            props.insert("compute_capability_minor".to_string(), self.device_info.compute_capability_minor.into_py(py));
            props.insert("multiprocessor_count".to_string(), self.device_info.multiprocessor_count.into_py(py));
            props.insert("max_threads_per_block".to_string(), self.device_info.max_threads_per_block.into_py(py));
            props.insert("warp_size".to_string(), self.device_info.warp_size.into_py(py));
            props.insert("concurrent_kernels".to_string(), self.device_info.concurrent_kernels.into_py(py));
            props.insert("ecc_enabled".to_string(), self.device_info.ecc_enabled.into_py(py));
            props.insert("uuid".to_string(), self.device_info.uuid.clone().into_py(py));
            
            Ok(props)
        })
    }
    
    fn __str__(&self) -> String {
        format!(
            "NetworkGPUDevice(id={}, name='{}', memory={:.1}GB)",
            self.device_info.device_id,
            self.device_info.name,
            self.device_info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        )
    }
    
    fn __repr__(&self) -> String {
        format!(
            "NetworkGPUDevice(device_id={}, name='{}', total_memory={}, free_memory={})",
            self.device_info.device_id,
            self.device_info.name,
            self.device_info.total_memory,
            self.device_info.free_memory
        )
    }
}

impl PyNetworkGPUDevice {
    pub fn from_device_info(device_info: DeviceInfo, client: Arc<NetworkGPUClient>) -> Self {
        Self { device_info, client }
    }
}