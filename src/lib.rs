use pyo3::prelude::*;
mod basic;
mod bipartite;
mod grid;
mod umap;

/// A Python module implemented in Rust.
#[pymodule]
fn _ilayoutx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(basic::line, m)?)?;
    m.add_function(wrap_pyfunction!(basic::circle, m)?)?;
    m.add_function(wrap_pyfunction!(basic::random, m)?)?;
    m.add_function(wrap_pyfunction!(basic::shell, m)?)?;
    m.add_function(wrap_pyfunction!(basic::spiral, m)?)?;
    m.add_function(wrap_pyfunction!(bipartite::bipartite, m)?)?;
    m.add_function(wrap_pyfunction!(grid::square, m)?)?;
    m.add_function(wrap_pyfunction!(grid::triangle, m)?)?;
    m.add_function(wrap_pyfunction!(umap::_umap_apply_forces, m)?)?;

    Ok(())
}
