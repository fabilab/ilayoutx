# ilayoutx

Compute fast network layouts. Intended as the upstream companion for [iplotx](https://github.com/fabilab/iplotx).

**NOTE**: This software is pre-alpha quality. There are two layouts working, and they are almost trivial. And yet.

## Installation
```bash
pip install ilayoutx
```

(not working yet)

## Layouts
- line
- circle
- bipartite
- random
- shell
- spiral (WIP)

## Rationale
The layout code is in Rust and exposed to Python via the amazing [PyO3](https://pyo3.rs/), with the goal to combine speed (by the machine) with comfort (for the user).

I'm a rust beginner, please be kind when judging this codebase. Feel free to open an [issue](https://github.com/fabilab/ilayoutx/issues) if you have questions.

## Authors
Fabio Zanini (https://fabilab.org)
