from typing import (
    Optional,
)
from collections.abc import (
    Hashable,
)
import numpy as np
import pandas as pd

from ..ingest import (
    network_library,
    data_providers,
)
from ..utils import _format_initial_coords
from ilayoutx._ilayoutx import (
    random as random_rust,
)


class Grid:
    def __init__(
        self,
        coords: np.ndarray,
        mins: np.ndarray,
        maxs: np.ndarray,
        deltas: np.ndarray,
    ):
        self.coords = coords
        self.mins = mins
        self.maxs = maxs
        self.deltas = deltas
        self.nsteps = np.ceil((maxs - mins) / deltas).astype(int)

        self.nv = coords.shape[0]
        self.n = 0
        self.added = np.zeros(self.nv, dtype=bool)
        self.center_n = np.zeros(2, dtype=np.float64)

    @property
    def center(self) -> np.ndarray:
        return self.center_n / self.n

    def add(
        self,
        indices: np.ndarray | int,
        coords: np.ndarray,
    ):
        self.coords[indices] = coords
        self.added[indices] = True
        if np.isscalar(indices):
            self.center_n += coords
            self.n += 1
        else:
            self.center_n += coords.sum(axis=0)
            self.n += len(indices)

    def get_cells(
        self,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Get the cells for the given indices."""
        coords = self.coords[indices]

        cells = np.floor((coords - self.mins) / self.deltas).astype(int)
        return cells

    def get_idx_neighboring_cells(
        self,
        indices: np.ndarray,
        cell: tuple[int, int] | np.ndarray,
        left: bool = True,
        right: bool = True,
        top: bool = True,
        bottom: bool = True,
    ) -> np.ndarray:
        """Determine which of the indices are in a cell neighboring the given cell.


        NOTE: I think igraph only considers the cell itself, right, and top. Not
        sure what the reason might be.
        """
        cells_indices = self.get_cells(indices)

        # Start with the cell itself
        samecell_idx = (cells_indices == cell).all(axis=1)

        # Neighboring cells
        if right:
            samecell_idx |= (cells_indices == cell + [1, 0]).all(axis=1)
        if top:
            samecell_idx |= (cells_indices == cell + [0, 1]).all(axis=1)
        if left:
            samecell_idx |= (cells_indices == cell + [-1, 0]).all(axis=1)
        if bottom:
            samecell_idx |= (cells_indices == cell + [0, -1]).all(axis=1)

        return indices[samecell_idx]


def large_graph_layout(
    network,
    initial_coords: Optional[
        dict[Hashable, tuple[float, float] | list[float]]
        | list[list[float] | tuple[float, float]]
        | np.ndarray
        | pd.DataFrame
    ] = None,
    center: Optional[tuple[float, float]] = (0, 0),
    etol: float = 1e-6,
    max_iter: int = 1000,
    root: Hashable = None,
    seed: Optional[int] = None,
    inplace: bool = True,
) -> pd.DataFrame:
    """Large graph layout (adapted) for connected networks.

    Parameters:
        network: The network to layout.
        initial_coords: Initial coordinates for the nodes. See also the "inplace" parameter.
        center: The center of the layout.
        etol: Gradient sum of spring forces must be larger than etol before successful termination.
        max_iter: Max iterations before termination of the algorithm.
        seed: A random seed to use.
        inplace: If True and the initial coordinates are a numpy array of dtype np.float64,
            that array will be recycled for the output and will be changed in place.
    Returns:
        The layout of the network.

    NOTE: This algorithm is not currently working on graphs that are not connected.

    NOTE: The algorithm used herein is an adaptation of the original LGL algorithm as conceived
    by the igraph team in igraph 0.2.
    """

    nl = network_library(network)
    provider = data_providers[nl](network)

    index = provider.vertices()
    nv = provider.number_of_vertices()

    if nv == 0:
        return pd.DataFrame(columns=["x", "y"])

    if nv == 1:
        coords = np.array([[0.0, 0.0]], dtype=np.float64)
    else:
        # Parameters
        area = nv**2
        cell_size = np.sqrt(nv)
        max_delta = 1.0 * nv
        area_side = np.sqrt(area / np.pi)
        cell_size = int(np.ceil(np.sqrt(nv)))

        # Compute minimum spanning tree. For now (as igraph), we ignore weights
        # and therefore any spanning tree via bfs is fine. If we want to use
        # weights, the issue is mostly getting the different providers' API
        # to align on an output format... just tedious that's all.
        if root is None:
            root_idx = np.random.randint(nv)
        else:
            root_idx = index.index(root)
        tree_dict = provider.bfs(root_idx)
        vertices_bfs = tree_dict["vertices"]
        parents = tree_dict["parents"]
        layer_switch = tree_dict["layer_switch"]
        del tree_dict

        nlayers = len(layer_switch) - 1
        harmonic_sum = (1.0 / np.linspace(1, nlayers - 1, nlayers - 1)).sum()

        # Force parameters
        force_k = np.sqrt(area / nv)
        repel_saddle = area * nv
        scon = np.sqrt(area / np.pi) / harmonic_sum

        # Default is a random layout scaled by the area of the grid
        initial_coords = _format_initial_coords(
            initial_coords,
            index=index,
            fallback=lambda: area_side * random_rust(nv, seed=seed),
            inplace=inplace,
        )

        coords = initial_coords

        grid = Grid(
            coords,
            mins=np.array([-area_side, -area_side], dtype=np.float64),
            maxs=np.array([area_side, area_side], dtype=np.float64),
            deltas=np.array([cell_size, cell_size], dtype=np.float64),
        )

        force = np.zeros((nv, 2), dtype=np.float64)

        # Place root
        grid.add(
            root_idx,
            np.zeros(2),
        )

        # Iterate over layers
        for ilayer in range(1, nlayers):
            max_delta = 1.0 + etol
            edges_samecell = []

            # 1. Place nodes in layer in a circle
            jprev, jcurr = layer_switch[ilayer - 1], layer_switch[ilayer]
            vertices_layer = vertices_bfs[jprev:jcurr]
            parents_layer = parents[jprev:jcurr]

            impulse = coords[vertices_layer] - coords[parents_layer]
            impulse /= np.linalg.norm(impulse + 1e-10, axis=1)[:, np.newaxis]

            # Add to the layout the next layer (i.e. starting from the children of root when ilayer == 1)
            cell_vertices = grid.get_cells(vertices_layer)
            for vertex_idx, imp, cell_vertex in zip(
                vertices_layer, impulse, cell_vertices
            ):
                children_idx = vertices_bfs[parents == vertex_idx]
                # Children of the root are spread evenly in a circle,
                # for the later layers use rotationally symmetric random sampling via Gaussians
                if ilayer == 1:
                    thetas = np.linspace(
                        0, 2 * np.pi, len(children_idx), endpoint=False
                    )
                    r = np.zeros((len(children_idx), 2), dtype=np.float64)
                    r[:, 0] = np.cos(thetas)
                    r[:, 1] = np.sin(thetas)
                    del thetas
                else:
                    r = np.random.normal(
                        loc=0.0, scale=1.0, size=(len(children_idx), 2)
                    )
                    r /= np.linalg.norm(r + 1e-10, axis=1)[:, np.newaxis]
                r *= scon / ilayer

                # Center + parent + nomalised parent impulse + rotational randomness
                grid.add(children_idx, grid.center + coords[vertex_idx] + imp + r)

                # 2. Record which of these children are within the same cell as their parent
                #    NOTE: This obviously makes a mess when the two nodes are right across a cell boundary,
                #    since the algorithm seems to ignore this case. I guess it's quantisation, baby.
                samecell_idx = grid.get_idx_neighboring_cells(
                    children_idx,
                    cell_vertex,
                )
                for child_idx in samecell_idx:
                    edges_samecell.append((vertex_idx, child_idx))

            edges_samecell = np.array(edges_samecell)

            # 3. Compute forces along those vetted edges
            maxchange = etol + 1
            # Iterate forces for this one layer
            for niter in range(1, max_iter + 1):
                print(niter, max_iter, maxchange, etol)

                # Learning rate (decreasing 1 -> 0, decelerating)
                eta = ((max_iter - niter) / max_iter) ** 1.5
                t = max_delta * eta

                force[:] = 0
                maxchange = 0.0

                # Attract along vetted, cross-layer edges, independent of grid cells
                if len(edges_samecell) > 0:
                    delta_edges = (
                        coords[edges_samecell[:, 1]] - coords[edges_samecell[:, 0]]
                    )
                    dist = np.linalg.norm(delta_edges)
                    nonoverlap_idx = dist > 0
                    # NOTE: This is a cheap way to zero contribution from
                    # overlapping vertices without changing edges_samecell,
                    # which should remain the same across the while loop
                    delta_edges[~nonoverlap_idx] = 0
                    force_abs = dist * dist / force_k
                    np.add.at(force, edges_samecell[:, 0], force_abs * delta_edges)
                    np.add.at(force, edges_samecell[:, 1], -force_abs * delta_edges)

                # Repel anything within the cell
                coords_added = grid.coords[grid.added]
                cells_added = grid.get_cells(grid.added)
                tmp = pd.DataFrame(cells_added, columns=["cell_x", "cell_y"])
                tmp["idx_added"] = np.arange(len(tmp))
                gby = tmp.groupby(["cell_x", "cell_y"])
                # FIXME: figure out a bit better what vertices are considered here
                for (idx_x, idx_y), vertices_cell in gby:
                    ncell = len(vertices_cell)
                    if ncell < 2:
                        continue
                    idx_cell = vertices_cell["idx_added"].values
                    coords_cell = coords_added[idx_cell]
                    delta_cell = (
                        coords_cell[:, np.newaxis, :] - coords_cell[np.newaxis, :, :]
                    )
                    dist2_cell = np.sum(delta_cell**2, axis=-1)
                    dist2_cell = np.maximum(dist2_cell, etol * etol)
                    delta_cell /= np.sqrt(dist2_cell)[:, :, np.newaxis]
                    force_abs = force_k**2 * (
                        1.0 / np.sqrt(dist2_cell) - dist2_cell / repel_saddle
                    )
                    # Remove self-repulsion
                    # NOTE: we could also exclude whenever mg0 == mg1, which are the
                    # self-repulsions, but that would require allocating new arrays
                    # that are of the same order of magnitude in size as these,
                    # hence less efficient than adding a small number of zeros
                    force_abs[np.arange(ncell), np.arange(ncell)] = 0.0

                    # Add force onto each vertex, in opposite directions
                    mg0, mg1 = np.meshgrid(idx_cell, idx_cell)
                    delta_cell = delta_cell.reshape((ncell * ncell, 2))
                    mg0 = mg0.reshape(ncell * ncell)
                    mg1 = mg1.reshape(ncell * ncell)
                    force_abs = force_abs.ravel()
                    # NOTE: because the matrices are symmetric and the diagonal
                    # is not useful but also not a problem (it's zero and operations
                    # are addition and subtraction), we can take the top half of
                    # these matrices after raveling, sacrifice the central diagonal
                    # element if ncell is odd and keep it if even (its value is zero)
                    # and perform the add/subtract operation only once. The order of
                    # indices is kind of awkward, but they are there once and  only once
                    # NOTE: all these raveled matrices have the same length, ncell**2
                    delta_cell = delta_cell[: len(delta_cell) // 2]
                    mg0 = mg0[: len(mg0) // 2]
                    mg1 = mg1[: len(mg1) // 2]
                    force_abs = force_abs[: len(force_abs) // 2]
                    np.add.at(force, mg0, delta_cell * force_abs[:, None])
                    np.add.at(force, mg1, -delta_cell * force_abs[:, None])

                # Move the nodes, with force clipped in absolute value
                force_abs = np.linalg.norm(force, axis=1)
                idx_exceed = force_abs > t
                force[idx_exceed] *= t / force_abs[idx_exceed, None]
                coords[:] += force

                # Record max change
                # NOTE: the igraph implementation does not include an absolute value here,
                # that looks kind of sus
                maxchange = max(maxchange, force.max())

                # Housekeeping
                if maxchange <= etol:
                    break

    coords += np.array(center, dtype=np.float64)

    layout = pd.DataFrame(coords, index=index, columns=["x", "y"])
    return layout
