use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyclass]
pub struct LinearLayer {
    weight: Py<PyAny>,
    bias: Option<Py<PyAny>>,
}

#[pymethods]
impl LinearLayer {
    #[new]
    pub fn new(py: Python, in_features: usize, out_features: usize, bias: bool) -> PyResult<Self> {
        let torch = py.import("torch")?;

        let weight_shape = PyTuple::new(py, [out_features, in_features])?;
        let weight = torch.call_method1("randn", (weight_shape,))?;
        let weight = weight.call_method1("requires_grad_", (true,))?;

        let bias_obj = if bias {
            let bias_shape = PyTuple::new(py, [out_features])?;
            let b = torch.call_method1("randn", (bias_shape,))?;
            let b = b.call_method1("requires_grad_", (true,))?;
            Some(b.unbind())
        } else {
            None
        };

        Ok(LinearLayer {
            weight: weight.unbind(),
            bias: bias_obj,
        })
    }

    pub fn forward(&self, py: Python, input: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let nn_functional = py.import("torch.nn.functional")?;

        let weight_bound = self.weight.bind(py);
        let bias_bound = self.bias.as_ref().map(|b| b.bind(py));

        let result = match bias_bound {
            Some(bias) => nn_functional.call_method1("linear", (&input, &weight_bound, &bias))?,
            None => nn_functional.call_method1("linear", (&input, &weight_bound, py.None()))?,
        };

        Ok(result.into_any().unbind())
    }

    #[getter]
    pub fn weight(&self, py: Python) -> PyObject {
        self.weight.clone_ref(py)
    }

    #[getter]
    pub fn bias(&self, py: Python) -> Option<PyObject> {
        self.bias.as_ref().map(|b| b.clone_ref(py))
    }

    pub fn parameters(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let mut params = vec![self.weight.clone_ref(py)];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone_ref(py));
        }
        Ok(params)
    }
}

#[pyclass]
#[derive(Default)]
pub struct ReLULayer;

#[pymethods]
impl ReLULayer {
    #[new]
    pub fn new() -> Self {
        ReLULayer
    }

    pub fn forward(&self, py: Python, input: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let torch_nn_functional = py.import("torch.nn.functional")?;
        let result = torch_nn_functional.call_method1("relu", (&input,))?;
        Ok(result.into_any().unbind())
    }
}

#[pyclass]
pub struct DropoutLayer {
    p: f64,
    training: bool,
}

#[pymethods]
impl DropoutLayer {
    #[new]
    pub fn new(p: f64) -> Self {
        DropoutLayer { p, training: true }
    }

    pub fn forward(&self, py: Python, input: Bound<'_, PyAny>) -> PyResult<PyObject> {
        let torch_nn_functional = py.import("torch.nn.functional")?;
        let result = torch_nn_functional.call_method1("dropout", (&input, self.p, self.training))?;
        Ok(result.into_any().unbind())
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    #[getter]
    pub fn training(&self) -> bool {
        self.training
    }
}

pub fn create_linear_layer(
    py: Python,
    in_features: usize,
    out_features: usize,
    bias: bool,
) -> PyResult<LinearLayer> {
    LinearLayer::new(py, in_features, out_features, bias)
}

pub fn create_relu_layer() -> ReLULayer {
    ReLULayer::new()
}

pub fn create_dropout_layer(p: f64) -> DropoutLayer {
    DropoutLayer::new(p)
}
