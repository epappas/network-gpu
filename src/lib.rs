use pyo3::prelude::*;

#[pyfunction]
fn add_tensors(py: Python, a: Bound<'_, PyAny>, b: Bound<'_, PyAny>) -> PyResult<PyObject> {
    tensor_ops::add_tensors(py, a, b)
}

#[pyfunction]
fn tensor_info(
    py: Python,
    tensor: Bound<'_, PyAny>,
) -> PyResult<std::collections::HashMap<String, PyObject>> {
    tensor_ops::tensor_info(py, tensor)
}

#[pyfunction]
fn multiply_tensors(py: Python, a: Bound<'_, PyAny>, b: Bound<'_, PyAny>) -> PyResult<PyObject> {
    tensor_ops::multiply_tensors(py, a, b)
}

#[pyfunction]
fn matmul_tensors(py: Python, a: Bound<'_, PyAny>, b: Bound<'_, PyAny>) -> PyResult<PyObject> {
    tensor_ops::matmul_tensors(py, a, b)
}

#[pyfunction]
#[pyo3(signature = (tensor, dim=None))]
fn tensor_sum(py: Python, tensor: Bound<'_, PyAny>, dim: Option<i32>) -> PyResult<PyObject> {
    tensor_ops::tensor_sum(py, tensor, dim)
}

#[pyfunction]
#[pyo3(signature = (tensor, dim=None))]
fn tensor_mean(py: Python, tensor: Bound<'_, PyAny>, dim: Option<i32>) -> PyResult<PyObject> {
    tensor_ops::tensor_mean(py, tensor, dim)
}

#[pyfunction]
fn create_linear_layer(
    py: Python,
    in_features: usize,
    out_features: usize,
    bias: bool,
) -> PyResult<neural_layers::LinearLayer> {
    neural_layers::create_linear_layer(py, in_features, out_features, bias)
}

#[pyfunction]
fn create_relu_layer() -> neural_layers::ReLULayer {
    neural_layers::create_relu_layer()
}

#[pyfunction]
fn create_dropout_layer(p: f64) -> neural_layers::DropoutLayer {
    neural_layers::create_dropout_layer(p)
}

#[pymodule]
fn _network_gpu(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Tensor operations
    m.add_function(wrap_pyfunction!(add_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_info, m)?)?;
    m.add_function(wrap_pyfunction!(multiply_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_sum, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_mean, m)?)?;

    // Neural network layers
    m.add_function(wrap_pyfunction!(create_linear_layer, m)?)?;
    m.add_function(wrap_pyfunction!(create_relu_layer, m)?)?;
    m.add_function(wrap_pyfunction!(create_dropout_layer, m)?)?;
    m.add_class::<neural_layers::LinearLayer>()?;
    m.add_class::<neural_layers::ReLULayer>()?;
    m.add_class::<neural_layers::DropoutLayer>()?;

    Ok(())
}
