use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::types::PyList;

/// Line layout, any angle theta in degrees
/// 
/// Parameters:
///     n (int): The number of vertices.
///     theta (float): The angle of the line in degrees.
/// Returns:
///     A list of tuples, each containing the x and y coordinates of the vertices.
#[pyfunction]
#[pyo3(signature = (n, theta=0.0))]
fn line(py: Python<'_>, n: usize, theta: f64) -> PyResult<Bound<'_, PyList>> {
    let coords: Bound<'_, PyList>;
    let theta = theta.to_radians();
    coords = PyList::empty(py);
    for i in 0..n {
        let coordsi: Bound<'_, PyTuple>;
        let elements: Vec<f64> = vec![(i as f64) * theta.cos(), (i as f64) * theta.sin()];
        coordsi = PyTuple::new(py, elements)?;
        coords.append(coordsi)?;
    }
    Ok(coords)
}


/// Circle layout, starting vertex at any angle theta in degrees
///
/// Parameters:
///     n (int): The number of vertices.
///     radius (float): The radius of the circle.
///     theta (float): The angle of the starting vertex in degrees.
/// Returns:
///     A list of tuples, each containing the x and y coordinates of the vertices.
#[pyfunction]
#[pyo3(signature = (n, radius=1.0, theta=0.0))]
fn circle(py: Python<'_>, n: usize, radius: f64, theta: f64) -> PyResult<Bound<'_, PyList>> {
    let coords: Bound<'_, PyList>;
    let theta = theta.to_radians();
    let alpha : f64 = 2.0 * std::f64::consts::PI / n as f64;
    coords = PyList::empty(py);
    for i in 0..n {
        let coordsi: Bound<'_, PyTuple>;
        let elements: Vec<f64> = vec![
            radius * (theta + alpha * (i as f64)).cos(),
            radius * (theta + alpha * (i as f64)).sin(),
        ];
        coordsi = PyTuple::new(py, elements)?;
        coords.append(coordsi)?;
    }
    Ok(coords)
}



/// A Python module implemented in Rust.
#[pymodule]
fn ilayoutx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(line, m)?)?;
    m.add_function(wrap_pyfunction!(circle, m)?)?;

    Ok(())
}

