"""Convenience functions for changing the data representation format."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import sparse


def hye_list_to_binary_incidence(
    hye_list: list[tuple[int]], shape: tuple[int] | None = None
) -> sparse.coo_array:
    """Convert a list of hyperedges into a scipy sparse COO array.
    The hyperedges need to be list of integers, representing nodes, starting from 0.
    If no shape is provided, this is inferred from the hyperedge list as (N, E).
    N is the number of nodes, given by the maximum integer observed in the hyperedge
    list plus one (since the node index starts from 0).
    E is the number of hyperedges in the list.
    If not None, the shape can only specify a tuple (N', E') where N' is greater or
    equal than the N inferred from the hyperedge list, and E' is greater or equal than
    the number of hyperedges in the list.

    Parameters
    ----------
    hye_list: the list of hyperedges.
        Every hyperedge is represented as a tuple of integer nodes.
    shape: the shape of the adjacency matrix, passed to the array constructor.
        If None, it is inferred.

    Returns
    -------
    The binary adjacency matrix representing the hyperedges.
    """
    rows = []
    columns = []
    for j, hye in enumerate(hye_list):
        # If there are repeated nodes in the hyperedge, count them once
        set_hye = set(hye)
        rows.extend(list(set_hye))
        columns.extend([j] * len(set_hye))

    inferred_N = max(rows) + 1
    inferred_E = len(hye_list)
    if shape is not None:
        if shape[0] < inferred_N or shape[1] < inferred_E:
            raise ValueError(
                "Provided shape incompatible with configurations hyperedge list."
            )
    else:
        shape = (inferred_N, inferred_E)

    data = np.ones_like(rows)

    return sparse.coo_array((data, (rows, columns)), shape=shape, dtype=np.uint8)


def incidence_matrix_to_hye(B: sparse.spmatrix) -> Iterable[np.ndarray]:
    """Extract an iterable of hyperedges from an incidence matrix.

    Parameters
    ----------
    B: incidence matrix with shape (N, E).

    Returns
    -------
    An iterable of hyperedges, each one represented as a numpy array of nodes.
    """
    B = sparse.csc_matrix(B)

    # See scipy's csc matrix documentation for explanation on data format:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html
    indices = B.indices
    indptr = B.indptr

    for i in range(len(indptr) - 1):
        yield indices[indptr[i] : indptr[i + 1]]
