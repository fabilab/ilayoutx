use pyo3::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use numpy::{PyArrayMethods, PyArray2};

/// Kamada Kawai layout
///
/// Parameters:
///     n (int): The number of vertices.
/// Returns:
///     An n x 2 numpy array containing random x and y coordinates of the vertices.
#[pyfunction]
#[pyo3(name = "kamada_kawai")]
#[pyo3(signature = (n))]
pub fn layout(py: Python<'_>, n: usize) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let coords = PyArray2::<f64>::zeros(py, [n, 2], true);
    
    if n > 1 {
        

    }

    Ok(coords)
}
