use networkgpu_client::*;
use numpy::{PyArray1, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};
use std::sync::Arc;
use tokio::runtime::Runtime;

mod client;
mod tensor;
mod device;
mod error;
mod runtime;

use client::PyNetworkGPUClient;
use device::PyNetworkGPUDevice;
use error::PyClientError;
use runtime::RUNTIME;
use tensor::PyRemoteTensor;

#[pyfunction]
fn init_networkgpu(
    py: Python,
    server_urls: Vec<String>,
    max_connections: Option<usize>,
    timeout_seconds: Option<u64>,
) -> PyResult<PyNetworkGPUClient> {
    let config = networkgpu_client::ClientConfig {
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
    
    Ok(PyNetworkGPUClient::new(Arc::new(client)))
}

#[pyfunction]
fn register_device_backend() -> PyResult<()> {
    Ok(())
}

#[pyfunction]
fn create_tensor(
    py: Python,
    data: &PyArrayDyn<f32>,
    device: &str,
    requires_grad: Option<bool>,
) -> PyResult<PyRemoteTensor> {
    let device_parts: Vec<&str> = device.split(':').collect();
    if device_parts.len() != 2 || device_parts[0] != "networkgpu" {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Device must be in format 'networkgpu:device_id'"
        ));
    }
    
    let device_id: i32 = device_parts[1].parse()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid device ID"))?;
    
    // Get shape from numpy array
    let shape: Vec<i64> = data.shape().iter().map(|&x| x as i64).collect();
    
    // Get raw data
    let raw_data = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>()
        )
    };
    
    Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
        "Direct tensor creation not yet implemented. Use PyNetworkGPUClient.create_tensor instead."
    ))
}

#[pymodule]
fn networkgpu(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_networkgpu, m)?)?;
    m.add_function(wrap_pyfunction!(register_device_backend, m)?)?;
    m.add_function(wrap_pyfunction!(create_tensor, m)?)?;
    
    m.add_class::<PyNetworkGPUClient>()?;
    m.add_class::<PyNetworkGPUDevice>()?;
    m.add_class::<PyRemoteTensor>()?;
    m.add_class::<PyClientError>()?;
    
    Ok(())
}