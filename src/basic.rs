use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Line layout, any angle theta in degrees
///
/// Parameters:
///     n (int): The number of vertices.
///     theta (float): The angle of the line in degrees.
/// Returns:
///     Ah n x 2 array containing the x and y coordinates of the vertices.
#[pyfunction]
#[pyo3(signature = (n, theta=0.0))]
pub fn line(py: Python<'_>, n: usize, theta: f64) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let theta = theta.to_radians();
    let coords = PyArray2::<f64>::zeros(py, [n, 2], true);
    for i in 0..n {
        unsafe {
            *coords.get_mut([i, 0]).unwrap() = (i as f64) * theta.cos();
            *coords.get_mut([i, 1]).unwrap() = (i as f64) * theta.sin();
        }
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
///     Ah n x 2 array containing the x and y coordinates of the vertices.
#[pyfunction]
#[pyo3(signature = (n, radius=1.0, theta=0.0))]
pub fn circle(
    py: Python<'_>,
    n: usize,
    radius: f64,
    theta: f64,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let theta = theta.to_radians();
    let alpha: f64 = 2.0 * std::f64::consts::PI / n as f64;
    let coords = PyArray2::<f64>::zeros(py, [n, 2], true);
    for i in 0..n {
        unsafe {
            *coords.get_mut([i, 0]).unwrap() = radius * (theta + alpha * (i as f64)).cos();
            *coords.get_mut([i, 1]).unwrap() = radius * (theta + alpha * (i as f64)).sin();
        }
    }
    Ok(coords)
}

/// Random layout
///
/// Parameters:
///     n (int): The number of vertices.
///     xmin (float): Minimum x coordinate.
///     xmax (float): Maximum x coordinate.
///     ymin (float): Minimum y coordinate.
///     ymax (float): Maximum y coordinate.
///     seed (int | None): Random seed. If None, ask the Operating System for a random seed.
/// Returns:
///     An n x 2 numpy array containing random x and y coordinates of the vertices.
#[pyfunction]
#[pyo3(signature = (n, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, seed=std::option::Option::None))]
pub fn random(
    py: Python<'_>,
    n: usize,
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    seed: Option<usize>,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let mut rng: StdRng;
    if seed.is_none() {
        rng = StdRng::from_os_rng();
    } else {
        rng = StdRng::seed_from_u64(u64::try_from(seed.unwrap())?);
    }
    let coords = PyArray2::<f64>::zeros(py, [n, 2], true);
    for i in 0..n {
        unsafe {
            *coords.get_mut([i, 0]).unwrap() = rng.random_range(xmin..xmax);
            *coords.get_mut([i, 1]).unwrap() = rng.random_range(ymin..ymax);
        }
    }
    Ok(coords)
}

/// Shell layout
///
/// Parameters:
///     nlist (list of integers): Each list contains the vertices to be laid out in that shell,
///         starting from the innermost shell. If the first shell has only one vertex, it
///         will be placed at the origin.
///     radius (float): The radius of the outermost shell.
///     center (pair of floats): The center of the layout.
///     theta (float): The angle of the first shell in degrees.
/// Returns:
///     An n x 2 numpy array containing random x and y coordinates of the vertices.
#[pyfunction]
#[pyo3(signature = (nlist, radius=1.0, center=(0.0, 0.0), theta=0.0))]
pub fn shell(
    py: Python<'_>,
    nlist: Vec<usize>,
    radius: f64,
    center: (f64, f64),
    theta: f64,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let theta = theta.to_radians();
    let n = nlist.iter().sum();
    let nshells = nlist.len();
    let coords = PyArray2::<f64>::zeros(py, [n, 2], true);

    if n > 0 {
        let mut r: f64 = 0.0;
        let mut i: usize = 0;
        let mut rshell: f64 = 0.0;
        let mut innermost = true;
        for nshell in nlist.iter() {
            if innermost {
                innermost = false;
                if *nshell > 1 {
                    rshell = radius / (nshells as f64);
                    r += rshell;
                } else if nshells > 1 {
                    rshell = radius / ((nshells - 1) as f64);
                } else {
                    // One shell, and that shell has one or zero elements
                    rshell = radius;
                }
            }
            let alpha = 2.0 * std::f64::consts::PI / *nshell as f64;
            for j in 0..*nshell {
                let angle = theta + alpha * j as f64;
                unsafe {
                    *coords.get_mut([i, 0]).unwrap() = center.0 + r * angle.cos();
                    *coords.get_mut([i, 1]).unwrap() = center.1 + r * angle.sin();
                }
                i += 1;
            }
            r += rshell;
        }
    }
    Ok(coords)
}

/// Spiral layout
///
/// Parameters:
///     n (int): The number of vertices.
///     slope (float): Radius increase per vertex (scaled). Lower values increase
///         the number of loops withing the same radius.
///     theta (float): The initial angle of the spiral in degrees.
///     exponent (float): The exponent of the spiral. 1.0 for Archimedean. For
///         logarithmic spirals, this parameter is the multiplier inside the exponential.
///
/// Returns:
///     An n x 2 numpy array containing random x and y coordinates of the vertices.
#[pyfunction]
#[pyo3(signature = (n, slope=1.0, theta=0.0, exponent=1.0))]
pub fn spiral(
    py: Python<'_>,
    n: usize,
    slope: f64,
    theta: f64,
    exponent: f64,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let theta = theta.to_radians();
    let coords = PyArray2::<f64>::zeros(py, [n, 2], true);

    let mut angle: f64 = theta;
    let mut r: f64;
    for i in 0..n {
        r = ((i + 1) as f64 * slope).powf(exponent);
        angle += 1.0 / r;
        unsafe {
            *coords.get_mut([i, 0]).unwrap() = r * angle.cos();
            *coords.get_mut([i, 1]).unwrap() = r * angle.sin();
        }
    }
    Ok(coords)
}
