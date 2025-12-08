use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Bipartite layout, two sets of vertices at any distance and angle theta in degrees
///
/// Parameters:
///     n1 (int): The number of vertices in the first set.
///     n2 (int): The number of vertices in the second set.
///     distance (float): The distance between the two sets of vertices.
///     theta (float): The angle of the line connecting the two sets in degrees.
/// Returns:
///     An (n1 + n2) x 2 numpy array, where the first n1 rows are the coordinates of the first set
///     and the next n2 rows are the coordinates of the second set.
#[pyfunction]
#[pyo3(signature = (n1, n2, distance=1.0, theta=0.0))]
pub fn bipartite(
    py: Python<'_>,
    n1: usize,
    n2: usize,
    distance: f64,
    theta: f64,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let theta = theta.to_radians();
    let ntot = n1 + n2;
    let coords = PyArray2::<f64>::zeros(py, [ntot, 2], true);
    for i in 0..n1 {
        unsafe {
            *coords.get_mut([i, 0]).unwrap() = (i as f64) * theta.sin();
            *coords.get_mut([i, 1]).unwrap() = (i as f64) * theta.cos();
        }
    }
    for i in 0..n2 {
        unsafe {
            *coords.get_mut([n1 + i, 0]).unwrap() =
                distance * theta.cos() + (i as f64) * theta.sin();
            *coords.get_mut([n1 + i, 1]).unwrap() =
                -distance * theta.sin() + (i as f64) * theta.cos();
        }
    }
    Ok(coords)
}
