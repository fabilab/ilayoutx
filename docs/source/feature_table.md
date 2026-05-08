# Layouts Feature Table

| Layout | Angle | Center/Position | Scaling/Radius | Random seed | Sizing | max iter | Init coords | Additional params* |
|-|-|-|-|-|-|-|-|-|
| {py:func}`Line <ilayoutx.layouts.line>`| ✓ | ✓ | ✓ | ✕ | ✕ | ✕ | ✕ | - |
| {py:func}`Circle <ilayoutx.layouts.circle>` | ✓ | ✓ | ✓| ✕| ✓ | ✕ | ✕ | - |
| {py:func}`Shell <ilayoutx.layouts.shell>` | ✓ | ✓ | ✓| ✕ | ✕ |✕ | ✕ | - |
| {py:func}`Spiral <ilayoutx.layouts.spiral>` | ✓ | ✓ | ✓| ✕ | ✕ |✕ | ✕ | slope, exponent |
| {py:func}`Random <ilayoutx.layouts.random>` | ✕ | ✕ | ✕| ✓| ✕ |✕ | ✕ | max_tries |
| {py:func}`Grid <ilayoutx.layouts.grid>` | ✕ | ✕ | ✓| ✕| ✕ |✕ | ✕ | width, shape, trim_even_rows (triangular) |
| {py:func}`Bipartite <ilayoutx.layouts.bipartite>` | ✕ | ✓ | ✕ | ✕| ✕ |✕ | ✕ | first partition, distance |
| {py:func}`Multipartite <ilayoutx.layouts.multipartite>` | ✕ | ✓ | ✕| ✕| ✕ |✕ | ✕ | - |
| {py:func}`Sugiyama <ilayoutx.layouts.sugiyama>` | ✕ | ✓ | ✕ | ✕| ✕ |✕ | ✕ | first partition |
| {py:func}`Spring <ilayoutx.layouts.spring>` | ✕ | ✕ | ✓ | ✓ | ✓ |✓ | ✓ | optimal_distance, gravity, fixed nodes, method (force/energy), exponent attraction/repulsion, etol|
| {py:func}`Kamada Kawai <ilayoutx.layouts.kamada_kawai>` | ✕ | ✕ | ✕ | ✓ | ✕ |✕ | ✕ | - |
| {py:func}`Arf <ilayoutx.layouts.arf>` | ✕ | ✓ | ✓ | ✓| ✕ |✓ | ✓ | etol, spring_strengh, dt (time step) |
| {py:func}`Forceatlas2 <ilayoutx.layouts.forceatlas2>`| ✕| ✓ | ✓| ✓ | ✓ |✓ | ✓ | jitter_tolerance, gravity + gtrong_gravity, distribution_action, mass, dissuade_hubs, linlog, etol |
| {py:func}`Graph Embedder <ilayoutx.layouts.graph_embedder>`| ✕| ✓ | ✓ | ✓ | ✓ | ✓| ✓ | etol, inplace |
| {py:func}`Large Graph Layout <ilayoutx.layouts.large_graph_layout>`| ✕ | ✓ | ✓| ✓| ✕ | ✓|✓ | inplace |
| {py:func}`Geometric <ilayoutx.layouts.geometric>` | ✕ | ✓ | ✕ | ✓| ✕ |✕ | ✕ | edge lengths, tol |
| {py:func}`Multidimensional scaling <ilayoutx.layouts.multidimensional_scaling>` | ✕ | ✓ | ✕ | ✕ | ✕ |✕ | ✕| distance_matrix, inplace, check_connectedness |
| {py:func}`UMAP <ilayoutx.layouts.umap>` | ✕ | ✕ | ✕ | ✕ | ✕ |✓ | ✓ | edge distance, edge weights, fixed, min_dist, spread, negative_sampling_rate, inplace, backend |

* See Layout documentation(./api/layouts.md) for exact requirements and further information about these params.