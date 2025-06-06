use networkgpu_client::{NetworkGPUClient, ClientConfig};
use numpy::{PyArray1, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use std::sync::Arc;

use crate::device::PyNetworkGPUDevice;
use crate::error::PyClientError;
use crate::runtime::RUNTIME;
use crate::tensor::PyRemoteTensor;

#[pyclass]
#[derive(Clone)]
pub struct PyNetworkGPUClient {
    client: Arc<NetworkGPUClient>,
}

#[pymethods]
impl PyNetworkGPUClient {
    #[new]
    fn new_py(
        server_urls: Vec<String>,
        max_connections: Option<usize>,
        timeout_seconds: Option<u64>,
    ) -> PyResult<Self> {
        let config = ClientConfig {
            connection: networkgpu_client::ConnectionConfig {
                max_connections_per_endpoint: max_connections.unwrap_or(4),
                request_timeout: std::time::Duration::from_secs(timeout_seconds.unwrap_or(300)),
                ..Default::default()
            },
            ..Default::default()
        };
        
        let client = RUNTIME.block_on(async {
            NetworkGPUClient::new(server_urls, config).await
        }).map_err(PyClientError::from)?;
        
        Ok(Self::new(Arc::new(client)))
    }
    
    /// List available GPU devices
    fn list_devices(&self) -> PyResult<Vec<PyNetworkGPUDevice>> {
        let devices = RUNTIME.block_on(async {
            self.client.list_devices().await
        }).map_err(PyClientError::from)?;
        
        Ok(devices.into_iter()
            .map(|device_info| PyNetworkGPUDevice::from_device_info(device_info, self.client.clone()))
            .collect())
    }
    
    fn get_device_info(&self, device_id: i32) -> PyResult<PyNetworkGPUDevice> {
        let device_info = RUNTIME.block_on(async {
            self.client.get_device_info(device_id).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyNetworkGPUDevice::from_device_info(device_info, self.client.clone()))
    }
    
    fn create_tensor(
        &self,
        shape: Vec<i64>,
        dtype: &str,
        device_id: i32,
        requires_grad: Option<bool>,
    ) -> PyResult<PyRemoteTensor> {
        let handle = RUNTIME.block_on(async {
            self.client.create_tensor(
                &shape,
                dtype,
                device_id,
                requires_grad.unwrap_or(false),
                None,
            ).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_handle(handle))
    }
    
    fn tensor_from_numpy(
        &self,
        py: Python,
        data: &PyArrayDyn<f32>,
        device_id: i32,
        requires_grad: Option<bool>,
    ) -> PyResult<PyRemoteTensor> {
        let shape: Vec<i64> = data.shape().iter().map(|&x| x as i64).collect();
        
        // Convert numpy data to bytes
        let raw_data = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>()
            )
        };
        
        let handle = RUNTIME.block_on(async {
            self.client.create_tensor(
                &shape,
                "float32",
                device_id,
                requires_grad.unwrap_or(false),
                Some(raw_data),
            ).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_handle(handle))
    }
    
    fn randn(&self, shape: Vec<i64>, device_id: i32) -> PyResult<PyRemoteTensor> {
        let tensor = RUNTIME.block_on(async {
            self.client.randn(&shape, device_id).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(tensor))
    }
    
    fn zeros(&self, shape: Vec<i64>, device_id: i32) -> PyResult<PyRemoteTensor> {
        let tensor = RUNTIME.block_on(async {
            self.client.zeros(&shape, device_id).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(tensor))
    }
    
    fn ones(&self, shape: Vec<i64>, device_id: i32) -> PyResult<PyRemoteTensor> {
        let tensor = RUNTIME.block_on(async {
            self.client.ones(&shape, device_id).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(tensor))
    }
    
    /// Synchronize all operations on a device
    fn synchronize(&self, device_id: i32) -> PyResult<()> {
        RUNTIME.block_on(async {
            self.client.synchronize_device(device_id).await
        }).map_err(PyClientError::from)?;
        
        Ok(())
    }
    
    fn health_check(&self) -> PyResult<bool> {
        let health = RUNTIME.block_on(async {
            self.client.health_check().await
        }).map_err(PyClientError::from)?;
        
        Ok(health.status == networkgpu_proto::health_check_response::Status::Serving as i32)
    }
    
    fn cache_stats(&self) -> PyResult<(usize, usize, usize, usize, usize)> {
        let stats = self.client.cache_stats();
        Ok((
            stats.tensor_cache_size,
            stats.tensor_cache_capacity,
            stats.device_cache_size,
            stats.device_cache_capacity,
            stats.memory_allocation_cache_size,
        ))
    }
    
    /// Clean up expired cache entries
    fn cleanup_cache(&self) {
        self.client.cleanup_cache();
    }
    
    fn __repr__(&self) -> String {
        "NetworkGPUClient()".to_string()
    }
}

impl PyNetworkGPUClient {
    pub fn new(client: Arc<NetworkGPUClient>) -> Self {
        Self { client }
    }
    
    pub fn get_client(&self) -> Arc<NetworkGPUClient> {
        self.client.clone()
    }
}