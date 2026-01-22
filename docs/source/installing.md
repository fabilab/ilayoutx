# Installing
```
pip install ilayoutx
```

If you want to use [numba](https://numba.pydata.org/) to accelerate and parallelise some layouts (notably UMAP):

```
pip install ilayoutx[fastmail]
```

If you would like to also install [iplotx](https://iplotx.readthedocs.io/en/latest/) for visualisation:

```
pip install ilayoutx[plot]
```

If you want to install everything including fastmath and plotting:

```
pip install ilayoutx[all]
```

## Quick Start
::::{tab-set}

:::{tab-item} igraph

```
>>> import igraph as ig
>>> import ilayoutx as ilx
>>> g = ig.Graph.Ring(5)
>>> layout = ilx.layouts.circle(g)
```


:::

:::{tab-item} networkx
```
>>> import networkx as nx
>>> import ilayoutx as ilx
>>> g = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
>>> layout = ilx.layouts.circle(g)
```

:::

::::

Either way, the result is the same:

```
>>> print(layout)
          x         y
0  1.000000  0.000000
1  0.309017  0.951057
2 -0.809017  0.587785
3 -0.809017 -0.587785
4  0.309017 -0.951057
```

If you want to visualise the layouted graph, you can use [iplotx](https://iplotx.readthedocs.io/en/latest/):

```
import iplotx as ipx
ipx.network(g, layout)
```

## Rationale
We believe graph **analysis**, graph **layouting**, and graph **visualisation** to be three separate tasks. `ilayoutx` currently focuses on layouting and the related task of edge routing.

## Citation
We have not yet written a publication for `ilayoutx`. If you would like to use `ilayoutx` for a publication, please contact us on our mailing list:

https://lists.sr.ht/~iosonofabio/ilayoutx-dev

## Contributing
Open an [issue on SourceHut](https://todo.sr.ht/~iosonofabio/ilayoutx) to request features, report bugs, or show intention in contributing.
