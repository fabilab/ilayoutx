use numpy::ndarray::{ArrayView2, ArrayViewMut2};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::Rng;

/// Clip both dimensions
fn clip_displacement(displacement: &mut [f32; 2], clip: f32) {
    for d in 0..2 {
        if displacement[d] < -clip {
            displacement[d] = -clip;
        } else if displacement[d] > clip {
            displacement[d] = clip;
        }
    }
}

/// Compute the move for a node connected by an edge according to UMAP attractive force
fn move_single_edge_nodes(pos1: [f32; 2], pos2: [f32; 2], a: f32, b: f32, clip: f32) -> [f32; 2] {
    let delta = [pos1[0] - pos2[0], pos1[1] - pos2[1]];
    let dist_squared = delta[0] * delta[0] + delta[1] * delta[1];

    if dist_squared < 1e-7 {
        return [0.0, 0.0];
    }
    let grad_coeff = -2.0 * (a * b * dist_squared.powf(b - 1.0)) / (a * dist_squared.powf(b) + 1.0);

    let mut displacement = [delta[0] * grad_coeff, delta[1] * grad_coeff];

    clip_displacement(&mut displacement, clip);
    displacement
}

fn _apply_repulsion_single(
    src: usize,
    dst_neg: usize,
    coords: &ArrayView2<'_, f32>,
    a: f32,
    b: f32,
    clip: f32,
) -> [f32; 2] {
    if dst_neg == src {
        return [0.0, 0.0];
    }

    let delta = [
        *coords.get([src, 0]).unwrap() - *coords.get([dst_neg, 0]).unwrap(),
        *coords.get([src, 1]).unwrap() - *coords.get([dst_neg, 1]).unwrap(),
    ];
    let dist_squared = delta[0] * delta[0] + delta[1] * delta[1];

    // Do not pop nodes that are EXACTLY on top of each other. This is ok since negative
    // sampling happens multiple times per epoch and in general this node might have
    // multiple edges each epoch. For very pathological graphs this might become an
    // issue.
    if dist_squared < 1e-14 {
        return [0.0, 0.0];
    }

    let grad_coeff = 2.0 * b / ((1e-3 + dist_squared) * (a * dist_squared.powf(b) + 1.0));

    if (-1e-14 < grad_coeff) & (grad_coeff < 1e-14) {
        return [0.0, 0.0];
    }

    let mut displacement_neg = [delta[0] * grad_coeff, delta[1] * grad_coeff];

    clip_displacement(&mut displacement_neg, clip);
    displacement_neg
}

// NOTE: The number of edges is the largest factor for runtime complexity, so this is the unit
// function in that respect. Ideally we would like to parallelise this one function call for
// clear benefits. In practice, the result does depend on the race condition in accessing
// coords. numba has a special "reduce" operation for this that essentially semaphores access to
// that one variable but lets the rest of the function go unrestricted. Not sure if that can be
// done esily in Rust.
fn _apply_forces_single_edge(
    edges: &ArrayView2<'_, i64>,
    coords: &mut ArrayViewMut2<'_, f32>,
    a: f32,
    b: f32,
    alpha: f32,
    clip: f32,
    negative_sampling_rate: usize,
    n: usize,
    i: usize,
) {
    let mut rng = rand::rng();

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
    *coords.get_mut([src, 0]).unwrap() += alpha * displacement[0];
    *coords.get_mut([src, 1]).unwrap() += alpha * displacement[1];

    // This is not a mapping to an existing embedding, so move the target node as well
    *coords.get_mut([dst, 0]).unwrap() -= alpha * displacement[0];
    *coords.get_mut([dst, 1]).unwrap() -= alpha * displacement[1];

    // NOTE: Parallelising this bit is useless and slows things down
    // The commented code here below is for educational purposes only
    //let rn_neg: Vec<usize> = (0..negative_sampling_rate)
    //    .map(|_| rng.random_range(0..n))
    //    .collect();
    //let displacement_neg: [f32; 2] = rn_neg
    //    .into_par_iter()
    //    .map(|dst_neg| {
    //        _apply_repulsion_single(src, dst_neg, &coords.view(), a, b, clip) as [f32; 2]
    //    })
    //    .reduce(|| [0.0, 0.0], |acc, x| [acc[0] + x[0], acc[1] + x[1]]);

    let displacement_neg = (0..negative_sampling_rate)
        .map(|_| {
            let dst_neg = rng.random_range(0..n);
            _apply_repulsion_single(src, dst_neg, &coords.view(), a, b, clip) as [f32; 2]
        })
        .fold([0.0, 0.0], |acc, x| [acc[0] + x[0], acc[1] + x[1]]);

    // Move the source node away from the negative sample
    *coords.get_mut([src, 0]).unwrap() += alpha * displacement_neg[0];
    *coords.get_mut([src, 1]).unwrap() += alpha * displacement_neg[1];
}

/// Compute attractive forces for UMAP
///
/// Parameters:
///     edges (numpy.ndarray): An array of shape (m, 2) containing the target indices of the edges.
///     coords (numpy.ndarray): An array of shape (n, 2) containing the component index of each
///     vertex.
///     a (float): The 'a' parameter for the UMAP attractive force function.
///     b (float): The 'b' parameter for the UMAP attractive force function.
///     alpha (float): The learning rate for the attractive forces.
///     clip (float): The maximum distance at which to compute attractive forces.
/// Returns:
///     Nothing
#[pyfunction]
#[pyo3(signature = (edges, coords, ab, alpha, clip=4.0, negative_sampling_rate=5))]
pub fn _umap_apply_forces(
    edges: PyReadonlyArray2<'_, i64>,
    coords: &Bound<'_, PyArray2<f32>>,
    ab: (f32, f32),
    alpha: f32,
    clip: f32,
    negative_sampling_rate: usize,
) {
    let a = ab.0;
    let b = ab.1;
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

    (0..m).for_each(|i| {
        _apply_forces_single_edge(
            &edges,
            &mut coords,
            a,
            b,
            alpha,
            clip,
            negative_sampling_rate,
            n,
            i,
        );
    });
}
