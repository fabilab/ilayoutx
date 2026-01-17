[![builds.sr.ht status](https://builds.sr.ht/~iosonofabio/ilayoutx.svg)](https://builds.sr.ht/~iosonofabio/ilayoutx?)
[![Github Actions](https://github.com/fabilab/ilayoutx/actions/workflows/CI.yml/badge.svg)](https://github.com/fabilab/ilayoutx/actions/workflows/CI.yml)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ilayoutx)](https://pypi.python.org/pypi/ilayoutx)
[![PyPI wheels](https://img.shields.io/pypi/wheel/ilayoutx.svg)](https://pypi.python.org/pypi/ilayoutx)
![Coverage](coverage-badge.svg)


# ilayoutx

Compute fast network layouts. Intended as the upstream companion for [iplotx](https://git.sr.ht/~iosonofabio/iplotx).

**NOTE**: This software is pre-alpha quality. The API is very much in flux, and the documentation is sparse. Use at your own risk.

## Installation
```bash
pip install ilayoutx
```

## Resources
 - **Issues**: https://todo.sr.ht/~iosonofabio/ilayoutx
 - **Mailing list**: https://lists.sr.ht/~iosonofabio/ilayoutx-dev
 - **Pull Requests**: This project prefers patches via the mailing list, however PRs on GitHub are currently accepted.

## Quickstart
```python
import networkx as nx
import ilayoutx as ilx

G = nx.circulant_graph(4, [1])
layout = ilx.layouts.multidimensional_scaling(G)
```


## Features
### Layouts
- **Shapes**:
  - line
  - circle (supports vertex sizes)
  - shell
  - spiral

- **Grid or lattice**:
  - square
  - triangular

- **Force-directed**:
  - spring (Fruchterman-Reingold)
  - ARF
  - Forceatlas2
  - Kamada-Kawai
  - GEM (graph embedder)
  - Geometric (from [netgraph](https://github.com/paulbrodersen/netgraph))
  - LGL (from [igraph](https://igraph.org/))

- **Directed acyclic graphs (DAGs)**:
  - Sugiyama including edge routing (only for directed graphs ATM).

- **Machine learning**:
  - UMAP (supports **arbitrary graphs**, not just knn graphs)

- **Other**:
  - bipartite
  - multipartite
  - random (supports vertex sizes)
  - multidimensional scaling (MDS)

### Packings
- Circular packing (via [circlify](github.com/elmotec/circlify/))
- Rectangular packing (via [rectangle-packer](https://github.com/Penlect/rectangle-packer/))

### Edge routing
`ilayoutx` includes routines to route edges in a visually pleasing way. This is generally tricky not only because
aesthetics are subjective, but also because the task is somewhat dependent on the level of zoom of the downstream
visualisation (intuitively, when zoomed out, things tend to look more crowded). Edge routing can be used in
[iplotx](https://git.sr.ht/~iosonofabio/iplotx) via the `waypoints` keyword argument of the `network` function.

The following edge routing algorithms are implemented:

- Sugiyama edge routing for DAGs.

## Wishlist
- **Tree-like**:
  - Reingold-Tilford

## Rationale
The layout code is in Rust and exposed to Python via the amazing [PyO3](https://pyo3.rs/), with the goal to combine speed (by the machine) with comfort (for the user).

I'm a rust beginner, please be kind when judging this codebase. Feel free to open an [issue](https://todo.sr.ht/~iosonofabio/ilayoutx) on SourceHut if you have questions.

## Authors
Fabio Zanini (https://fabilab.org)
