[![Github Actions](https://github.com/fabilab/iplotx/actions/workflows/test.yml/badge.svg)](https://github.com/fabilab/iplotx/actions/workflows/test.yml)
[![builds.sr.ht status](https://builds.sr.ht/~iosonofabio/ilayoutx.svg)](https://builds.sr.ht/~iosonofabio/ilayoutx?)
[![PyPI - Version](https://img.shields.io/pypi/v/ilayoutx)](https://pypi.org/project/ilayoutx/)


# ilayoutx

Compute fast network layouts. Intended as the upstream companion for [iplotx](https://github.com/fabilab/iplotx).

**NOTE**: This software is pre-alpha quality. The API is very much in flux, and the documentation is sparse. Use at your own risk.

## Installation
```bash
pip install ilayoutx
```

(not working yet)

## Layouts
- **Geometric**:
  - line
  - circle (supports vertex sizes)
  - shell
  - spiral (WIP)

- **Grid or lattice**:
  - square
  - triangular

- **Force-directed**:
  - spring aka Fruchterman-Reingold
  - ARF
  - Forceatlas2
  - Kamada-Kawai
  - GEM (graph embedder)
  - LGL (buggy)

- **Machine learning**:
  - UMAP: probably somewhat buggy

- **Tree-like**:
  - (TODO: Reingold-Tilford)
  - (TODO: Sugiyama)

- **Other**:
  - bipartite
  - random (supports vertex sizes)
  - multidimensional scaling (MDS)


## Issues
Issues are tracked on SourceHut: https://todo.sr.ht/~iosonofabio/ilayoutx

## Rationale
The layout code is in Rust and exposed to Python via the amazing [PyO3](https://pyo3.rs/), with the goal to combine speed (by the machine) with comfort (for the user).

I'm a rust beginner, please be kind when judging this codebase. Feel free to open an [issue](https://github.com/fabilab/ilayoutx/issues) if you have questions.

## Authors
Fabio Zanini (https://fabilab.org)
