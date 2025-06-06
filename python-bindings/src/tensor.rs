use networkgpu_client::{RemoteTensor, TensorHandle};
use numpy::{PyArray1, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

use crate::error::PyClientError;
use crate::runtime::RUNTIME;

#[pyclass]
#[derive(Clone)]
pub struct PyRemoteTensor {
    inner: Option<RemoteTensor>,
    handle: Option<TensorHandle>,
}

#[pymethods]
impl PyRemoteTensor {
    /// Get tensor shape
    #[getter]
    fn shape(&self) -> PyResult<Vec<i64>> {
        if let Some(ref tensor) = self.inner {
            Ok(tensor.shape.clone())
        } else if let Some(ref handle) = self.handle {
            let shape = RUNTIME.block_on(async {
                handle.shape().await
            });
            Ok(shape)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid tensor state"))
        }
    }
    
    /// Get tensor data type
    #[getter]
    fn dtype(&self) -> PyResult<String> {
        if let Some(ref tensor) = self.inner {
            Ok(tensor.dtype.clone())
        } else if let Some(ref handle) = self.handle {
            let dtype = RUNTIME.block_on(async {
                handle.dtype().await
            });
            Ok(dtype)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid tensor state"))
        }
    }
    
    /// Get device ID
    #[getter]
    fn device_id(&self) -> PyResult<i32> {
        if let Some(ref tensor) = self.inner {
            Ok(tensor.device_id)
        } else if let Some(ref handle) = self.handle {
            let device_id = RUNTIME.block_on(async {
                handle.device_id().await
            });
            Ok(device_id)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid tensor state"))
        }
    }
    
    /// Get tensor size in bytes
    #[getter]
    fn size_bytes(&self) -> PyResult<u64> {
        if let Some(ref handle) = self.handle {
            let size = RUNTIME.block_on(async {
                handle.size_bytes().await
            });
            Ok(size)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid tensor state"))
        }
    }
    
    /// Whether tensor requires gradients
    #[getter]
    fn requires_grad(&self) -> PyResult<bool> {
        if let Some(ref tensor) = self.inner {
            Ok(tensor.requires_grad)
        } else if let Some(ref handle) = self.handle {
            let requires_grad = RUNTIME.block_on(async {
                handle.requires_grad().await
            });
            Ok(requires_grad)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid tensor state"))
        }
    }
    
    /// Add two tensors
    fn add(&self, other: &PyRemoteTensor) -> PyResult<PyRemoteTensor> {
        let (self_tensor, other_tensor) = self.get_tensors(other)?;
        
        let result = RUNTIME.block_on(async {
            self_tensor.add(&other_tensor).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(result))
    }
    
    /// Subtract two tensors
    fn sub(&self, other: &PyRemoteTensor) -> PyResult<PyRemoteTensor> {
        let (self_tensor, other_tensor) = self.get_tensors(other)?;
        
        let result = RUNTIME.block_on(async {
            self_tensor.subtract(&other_tensor).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(result))
    }
    
    /// Multiply two tensors (element-wise)
    fn mul(&self, other: &PyRemoteTensor) -> PyResult<PyRemoteTensor> {
        let (self_tensor, other_tensor) = self.get_tensors(other)?;
        
        let result = RUNTIME.block_on(async {
            self_tensor.multiply(&other_tensor).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(result))
    }
    
    /// Divide two tensors (element-wise)
    fn div(&self, other: &PyRemoteTensor) -> PyResult<PyRemoteTensor> {
        let (self_tensor, other_tensor) = self.get_tensors(other)?;
        
        let result = RUNTIME.block_on(async {
            self_tensor.divide(&other_tensor).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(result))
    }
    
    /// Matrix multiplication
    fn matmul(&self, other: &PyRemoteTensor) -> PyResult<PyRemoteTensor> {
        let (self_tensor, other_tensor) = self.get_tensors(other)?;
        
        let result = RUNTIME.block_on(async {
            self_tensor.matmul(&other_tensor).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(result))
    }
    
    /// Transpose tensor
    fn transpose(&self, dim0: i64, dim1: i64) -> PyResult<PyRemoteTensor> {
        let tensor = self.get_tensor()?;
        
        let result = RUNTIME.block_on(async {
            tensor.transpose(dim0, dim1).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(result))
    }
    
    /// Reshape tensor
    fn reshape(&self, new_shape: Vec<i64>) -> PyResult<PyRemoteTensor> {
        let tensor = self.get_tensor()?;
        
        let result = RUNTIME.block_on(async {
            tensor.reshape(&new_shape).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(result))
    }
    
    /// Sum tensor along dimension
    fn sum(&self, dim: Option<i64>) -> PyResult<PyRemoteTensor> {
        let tensor = self.get_tensor()?;
        
        let result = RUNTIME.block_on(async {
            tensor.sum(dim).await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(result))
    }
    
    /// Apply sigmoid activation
    fn sigmoid(&self) -> PyResult<PyRemoteTensor> {
        let tensor = self.get_tensor()?;
        
        let result = RUNTIME.block_on(async {
            tensor.sigmoid().await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(result))
    }
    
    /// Apply ReLU activation
    fn relu(&self) -> PyResult<PyRemoteTensor> {
        let tensor = self.get_tensor()?;
        
        let result = RUNTIME.block_on(async {
            tensor.relu().await
        }).map_err(PyClientError::from)?;
        
        Ok(PyRemoteTensor::from_remote_tensor(result))
    }
    
    /// Copy tensor to CPU and return as numpy array
    fn cpu(&self, py: Python) -> PyResult<PyObject> {
        let data = if let Some(ref handle) = self.handle {
            RUNTIME.block_on(async {
                handle.get_data().await
            }).map_err(PyClientError::from)?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid tensor state"));
        };
        
        let shape = self.shape()?;
        let dtype = self.dtype()?;
        
        // Convert bytes back to the appropriate type
        match dtype.as_str() {
            "float32" => {
                let float_data: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const f32,
                        data.len() / std::mem::size_of::<f32>()
                    )
                };
                
                // Create numpy array with the correct shape
                let array = PyArray1::from_slice(py, float_data)
                    .reshape(shape.iter().map(|&x| x as usize).collect::<Vec<_>>())
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Reshape failed: {}", e)))?;
                
                Ok(array.to_object(py))
            }
            "int32" => {
                let int_data: &[i32] = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const i32,
                        data.len() / std::mem::size_of::<i32>()
                    )
                };
                
                let array = PyArray1::from_slice(py, int_data)
                    .reshape(shape.iter().map(|&x| x as usize).collect::<Vec<_>>())
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Reshape failed: {}", e)))?;
                
                Ok(array.to_object(py))
            }
            _ => {
                // For unsupported types, return raw bytes
                Ok(PyBytes::new(py, &data).to_object(py))
            }
        }
    }
    
    /// Copy data from numpy array to tensor
    fn copy_from_numpy(&self, py: Python, data: &PyArrayDyn<f32>) -> PyResult<()> {
        let raw_data = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>()
            )
        };
        
        if let Some(ref handle) = self.handle {
            RUNTIME.block_on(async {
                handle.set_data(raw_data).await
            }).map_err(PyClientError::from)?;
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid tensor state"));
        }
        
        Ok(())
    }
    
    /// Get tensor data as bytes
    fn to_bytes(&self, py: Python) -> PyResult<PyObject> {
        let data = if let Some(ref handle) = self.handle {
            RUNTIME.block_on(async {
                handle.get_data().await
            }).map_err(PyClientError::from)?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid tensor state"));
        };
        
        Ok(PyBytes::new(py, &data).to_object(py))
    }
    
    /// Python magic methods for operator overloading
    fn __add__(&self, other: &PyRemoteTensor) -> PyResult<PyRemoteTensor> {
        self.add(other)
    }
    
    fn __sub__(&self, other: &PyRemoteTensor) -> PyResult<PyRemoteTensor> {
        self.sub(other)
    }
    
    fn __mul__(&self, other: &PyRemoteTensor) -> PyResult<PyRemoteTensor> {
        self.mul(other)
    }
    
    fn __truediv__(&self, other: &PyRemoteTensor) -> PyResult<PyRemoteTensor> {
        self.div(other)
    }
    
    fn __matmul__(&self, other: &PyRemoteTensor) -> PyResult<PyRemoteTensor> {
        self.matmul(other)
    }
    
    fn __str__(&self) -> PyResult<String> {
        let shape = self.shape()?;
        let dtype = self.dtype()?;
        let device_id = self.device_id()?;
        
        Ok(format!(
            "NetworkGPUTensor(shape={:?}, dtype={}, device=networkgpu:{})",
            shape, dtype, device_id
        ))
    }
    
    fn __repr__(&self) -> PyResult<String> {
        self.__str__()
    }
}

impl PyRemoteTensor {
    pub fn from_remote_tensor(tensor: RemoteTensor) -> Self {
        Self {
            inner: Some(tensor),
            handle: None,
        }
    }
    
    pub fn from_handle(handle: TensorHandle) -> Self {
        Self {
            inner: None,
            handle: Some(handle),
        }
    }
    
    fn get_tensor(&self) -> PyResult<&RemoteTensor> {
        self.inner.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No RemoteTensor available"))
    }
    
    fn get_tensors(&self, other: &PyRemoteTensor) -> PyResult<(&RemoteTensor, &RemoteTensor)> {
        let self_tensor = self.get_tensor()?;
        let other_tensor = other.get_tensor()?;
        Ok((self_tensor, other_tensor))
    }
}