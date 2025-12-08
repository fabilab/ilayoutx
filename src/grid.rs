use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Square grid layout
///
/// Parameters:
///     n (int): The number of vertices.
///     width (int): The number of vertices in each row of the grid.
/// Returns:
///     An n x 2 numpy array containing the x and y coordinates of the vertices arranged in a grid.
#[pyfunction]
#[pyo3(name = "grid_square")]
#[pyo3(signature = (n, width))]
pub fn square(py: Python<'_>, n: usize, width: usize) -> PyResult<Bound<'_, PyArray2<f64>>> {
    if width == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Width must be greater than 0",
        ));
    }
    let coords = PyArray2::<f64>::zeros(py, [n, 2], true);

    let mut x: usize = 0;
    let mut y: usize = 0;
    for i in 0..n {
        unsafe {
            *coords.get_mut([i, 0]).unwrap() = x as f64;
            *coords.get_mut([i, 1]).unwrap() = y as f64;
        }
        if x == width - 1 {
            x = 0;
            y += 1;
        } else {
            x += 1;
        }
    }
    Ok(coords)
}

/// Triangular grid layout
///
/// Parameters:
///     n (int): The number of vertices.
///     width (int): The number of vertices in each odd row of the grid.
///     equal_rows (bool): If true, all rows have the same number of vertices.
///         If false, odd rows have one more vertex than even rows.
/// Returns:
///     An n x 2 numpy array containing the x and y coordinates of the vertices arranged in a
///     triangular grid.
#[pyfunction]
#[pyo3(name = "grid_triangle")]
#[pyo3(signature = (n, width, equal_rows=true))]
pub fn triangle(
    py: Python<'_>,
    n: usize,
    width: usize,
    equal_rows: bool,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    if width == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Width must be greater than 0",
        ));
    }
    let coords = PyArray2::<f64>::zeros(py, [n, 2], true);
    let half_sqrt3 = 0.5 * (3.0_f64).sqrt();

    let mut x: usize = 0;
    let mut y: usize = 0;
    for i in 0..n {
        let xoffset: f64 = 0.5 * ((y % 2) as f64);
        let xmax: usize = if equal_rows {
            width - 1
        } else {
            width - 1 - (y % 2)
        };
        unsafe {
            *coords.get_mut([i, 0]).unwrap() = (x as f64) + xoffset;
            *coords.get_mut([i, 1]).unwrap() = (y as f64) * half_sqrt3;
        }
        if x == xmax {
            x = 0;
            y += 1
        } else {
            x += 1;
        }
    }
    Ok(coords)
}
