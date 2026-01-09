[![builds.sr.ht status](https://builds.sr.ht/~iosonofabio/ilayoutx.svg)](https://builds.sr.ht/~iosonofabio/ilayoutx?)
[![Github Actions](https://github.com/fabilab/ilayoutx/actions/workflows/CI.yml/badge.svg)](https://github.com/fabilab/ilayoutx/actions/workflows/CI.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/ilayoutx)](https://pypi.org/project/ilayoutx/)
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

- **Directed acyclic graphs (DAGs)**:
  - Sugiyama including edge routing (only for directed graphs ATM).

- **Other**:
  - bipartite
  - multipartite
  - random (supports vertex sizes)
  - multidimensional scaling (MDS)

Some layout functions are written but not (well) tested, therefore to be considered experimental.

- **Force-directed**:
  - LGL (buggy)

- **Machine learning**:
  - UMAP: (buggy)

### Packings
- Circular packing (via [circlify](github.com/elmotec/circlify/))

Some packing functions are written but not (well) tested, therefore to be considered experimental.

- Rectangular packing (via [rectangle-packer](https://github.com/Penlect/rectangle-packer/))


## Wishlist
- **Tree-like**:
  - Reingold-Tilford

## Rationale
The layout code is in Rust and exposed to Python via the amazing [PyO3](https://pyo3.rs/), with the goal to combine speed (by the machine) with comfort (for the user).

I'm a rust beginner, please be kind when judging this codebase. Feel free to open an [issue](https://todo.sr.ht/~iosonofabio/ilayoutx) on SourceHut if you have questions.

## Authors
Fabio Zanini (https://fabilab.org)
