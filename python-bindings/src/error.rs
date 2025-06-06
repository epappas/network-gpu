use networkgpu_client::ClientError;
use pyo3::prelude::*;
use pyo3::{create_exception, PyErr};

// Create custom Python exception types
create_exception!(networkgpu, NetworkGPUError, pyo3::exceptions::PyException);
create_exception!(networkgpu, DeviceError, NetworkGPUError);
create_exception!(networkgpu, TensorError, NetworkGPUError);
create_exception!(networkgpu, ConnectionError, NetworkGPUError);

#[pyclass(extends=NetworkGPUError)]
#[derive(Debug, Clone)]
pub struct PyClientError {
    #[pyo3(get)]
    pub message: String,
    #[pyo3(get)]
    pub error_type: String,
}

#[pymethods]
impl PyClientError {
    #[new]
    fn new(message: String, error_type: String) -> Self {
        Self { message, error_type }
    }
    
    fn __str__(&self) -> String {
        format!("{}: {}", self.error_type, self.message)
    }
    
    fn __repr__(&self) -> String {
        format!("PyClientError(message='{}', error_type='{}')", self.message, self.error_type)
    }
}

impl From<ClientError> for PyErr {
    fn from(err: ClientError) -> Self {
        let (exception_type, error_type) = match &err {
            ClientError::ConnectionError(_) => (ConnectionError::new_err, "ConnectionError"),
            ClientError::DeviceOperationFailed(_) => (DeviceError::new_err, "DeviceError"),
            ClientError::TensorOperationFailed(_) => (TensorError::new_err, "TensorError"),
            ClientError::InvalidDeviceId(_) => (DeviceError::new_err, "InvalidDeviceError"),
            ClientError::InvalidTensorId(_) => (TensorError::new_err, "InvalidTensorError"),
            _ => (NetworkGPUError::new_err, "NetworkGPUError"),
        };
        
        exception_type(err.to_string())
    }
}

impl From<ClientError> for PyClientError {
    fn from(err: ClientError) -> Self {
        let error_type = match &err {
            ClientError::ConnectionError(_) => "ConnectionError",
            ClientError::DeviceOperationFailed(_) => "DeviceError",
            ClientError::TensorOperationFailed(_) => "TensorError",
            ClientError::InvalidDeviceId(_) => "InvalidDeviceError",
            ClientError::InvalidTensorId(_) => "InvalidTensorError",
            _ => "NetworkGPUError",
        };
        
        Self {
            message: err.to_string(),
            error_type: error_type.to_string(),
        }
    }
}