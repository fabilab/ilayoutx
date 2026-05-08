# Data Providers
Networks and trees can be represented in different data structures. For instance, in `igraph` nodes are ordered and numbered, while in `networkx` they are neither.

`ilayoutx` is designed to accept data of any kinds as long as it can understand how to
process it into a layout. This flexibility is achieved through **data providers**.

A **network data provider** is a class that satisfies the `NetworkDataProvider` protocol, i.e. it implements a few functions that help `ilayoutx` understand how to ingest
data.

See <project:api/providers.md> for the exact protocols.

## Creating a custom data provider
`ilayoutx` is able to seek new data providers at runtime, but they need to be registered in a specific way (keyword: `entry point`).

The expected entry point for a `NetworkDataProvider` is: `ilayoutx.network_data_providers`.

```{note}
  If you would like to make a library compatible with `ilayoutx` (by developing a provider), we are here to help! Just reach out on GitHub issues and we will try to help.
```
