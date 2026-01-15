use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;

/// Compute the move for a node connected by an edge according to UMAP attractive force
fn move_single_edge_nodes(pos1: [f32; 2], pos2: [f32; 2], a: f64, b: f64, clip: f64) -> [f32; 2] {
    let delta = [pos1[0] - pos2[0], pos1[1] - pos2[1]];
    let dist_squared = delta[0] * delta[0] + delta[1] * delta[1];

    if dist_squared < 1e-7 {
        return [0.0, 0.0];
    }
    let grad_coeff = -2.0 * (a as f32 * b as f32 * dist_squared.powf(b as f32 - 1.0))
        / ((a as f32 * dist_squared.powf(b as f32) + 1.0).powf(2.0));

    let mut displacement = [delta[0] * grad_coeff, delta[1] * grad_coeff];

    if displacement[0] < -clip as f32 {
        displacement[0] = -clip as f32;
    } else if displacement[0] > clip as f32 {
        displacement[0] = clip as f32;
    }
    displacement
}

/// Compute attractive forces for UMAP
///
/// Parameters:
///     edges (numpy.ndarray): An array of shape (m, 2) containing the target indices of the edges.
///     coords (numpy.ndarray): An array of shape (n, 2) containing the component index of each
///     vertex.
///     a (float): The 'a' parameter for the UMAP attractive force function.
///     b (float): The 'b' parameter for the UMAP attractive force function.
///     clip (float): The maximum distance at which to compute attractive forces.
/// Returns:
///     Nothing
#[pyfunction]
#[pyo3(signature = (edges, coords, a, b, clip=4.0))]
pub fn _umap_attractive_forces(
    edges: PyReadonlyArray2<'_, i64>,
    coords: &Bound<'_, PyArray2<f32>>,
    a: f64,
    b: f64,
    clip: f64,
) {
    let edges = edges.as_array();
    let m = edges.shape()[0];
    let m2 = edges.shape()[1];

    let mut coords = coords.readwrite();
    let mut coords = coords.as_array_mut();
    let n = coords.shape()[0];
    let n2 = coords.shape()[1];

    if (m2 != 2) | (n2 != 2) {
        panic!("Edges must be of shape (m, 2) and coords must be of shape (n, 2)");
    }

    for i in 0..m {
        let src = *edges.get([i, 0]).unwrap() as usize;
        let dst = *edges.get([i, 1]).unwrap() as usize;
        if (src >= n) | (dst >= n) {
            panic!("Edge index out of bounds");
        }

        let displacement = move_single_edge_nodes(
            [
                *coords.get([src, 0]).unwrap(),
                *coords.get([src, 1]).unwrap(),
            ],
            [
                *coords.get([dst, 0]).unwrap(),
                *coords.get([dst, 1]).unwrap(),
            ],
            a,
            b,
            clip,
        );

        // Move the source node
        *coords.get_mut([src, 0]).unwrap() += displacement[0];
        *coords.get_mut([src, 1]).unwrap() += displacement[1];

        // This is not a mapping to an existing embedding, so move the target node as well
        *coords.get_mut([dst, 0]).unwrap() -= displacement[0];
        *coords.get_mut([dst, 1]).unwrap() -= displacement[1];
    }
}
