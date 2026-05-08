# API Reference
`ilayoutx` is organised in three main submodules: **layouts**, **edge routing**, and **packing**. These three are not 100% separate: for instance, some layout functions might call routing or packing functions internally.

- [layouts](./api/layouts.md): functions for computing node positions in a graph.
- [routing](./api/routing.md): functions for computing edge paths in a graph.
- [packing](./api/packing.md): functions for packing multiple graphs together, typically the connected components of a disconnected graph.

`ilayoutx` is designed to be universally compatible with any input graph library through **network data providers**:

- [data providers](./api/providers.md): Data providers are classes that satisfy specific protocols to help `ilayoutx` understand how to ingest data from different libraries and data structures. If you want to make your library compatible with `ilayoutx`, you can develop a custom data provider and register it as an entry point.

- [feature table](feature_table.md): View a compact table of features available for each graph.