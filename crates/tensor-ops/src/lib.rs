use pyo3::prelude::*;
use std::collections::HashMap;

pub fn add_tensors(py: Python, a: Bound<'_, PyAny>, b: Bound<'_, PyAny>) -> PyResult<PyObject> {
    let torch = py.import("torch")?;
    let result = torch.call_method1("add", (&a, &b))?;
    Ok(result.into_any().unbind())
}

pub fn tensor_info(py: Python, tensor: Bound<'_, PyAny>) -> PyResult<HashMap<String, PyObject>> {
    let mut info = HashMap::new();
    
    let shape = tensor.getattr("shape")?;
    let dtype = tensor.getattr("dtype")?;
    let device = tensor.getattr("device")?;
    let requires_grad = tensor.getattr("requires_grad")?;
    
    info.insert("shape".to_string(), shape.into_any().unbind());
    info.insert("dtype".to_string(), dtype.into_any().unbind());
    info.insert("device".to_string(), device.into_any().unbind());
    info.insert("requires_grad".to_string(), requires_grad.into_any().unbind());
    
    Ok(info)
}

pub fn multiply_tensors(py: Python, a: Bound<'_, PyAny>, b: Bound<'_, PyAny>) -> PyResult<PyObject> {
    let torch = py.import("torch")?;
    let result = torch.call_method1("mul", (&a, &b))?;
    Ok(result.into_any().unbind())
}

pub fn matmul_tensors(py: Python, a: Bound<'_, PyAny>, b: Bound<'_, PyAny>) -> PyResult<PyObject> {
    let torch = py.import("torch")?;
    let result = torch.call_method1("matmul", (&a, &b))?;
    Ok(result.into_any().unbind())
}

pub fn tensor_sum(py: Python, tensor: Bound<'_, PyAny>, dim: Option<i32>) -> PyResult<PyObject> {
    let result = match dim {
        Some(d) => tensor.call_method1("sum", (d,))?,
        None => tensor.call_method0("sum")?,
    };
    Ok(result.into_any().unbind())
}

pub fn tensor_mean(py: Python, tensor: Bound<'_, PyAny>, dim: Option<i32>) -> PyResult<PyObject> {
    let result = match dim {
        Some(d) => tensor.call_method1("mean", (d,))?,
        None => tensor.call_method0("mean")?,
    };
    Ok(result.into_any().unbind())
}