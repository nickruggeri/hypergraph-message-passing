from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse

from src.data.conversion import hye_list_to_binary_incidence, incidence_matrix_to_hye
from src.data.representation.binary_hypergraph import BinaryHypergraph

# Define the default type of sparse matrix utilized for the incidence matrix inside the
# IncidenceHypergraph instances.
TYPE_INCIDENCE: sparse.spmatrix = sparse.csc_array


class IncidenceHypergraph(BinaryHypergraph):
    """Representation of a binary hypergraph via its incidence matrix.
    The incidence matrix B is of size N x E, with N number of nodes in the hypergraph
    and E number of hyperedges. For each hyperedge e, the column of B with index e
    contains ones for the nodes belonging to the hyperedge e, zeros for all other nodes.
    """

    def __init__(
        self,
        B: np.ndarray | sparse.spmatrix,
        sort_indices: bool = True,
    ):
        """
        Parameters
        ----------
        B: incidence matrix, of shape (N, E).
        sort_indices: sort the indices in the internal sparse matrix representation.
        """
        self.B = self._check_and_convert_incidence(B, sort_indices)
        self.N, self.E = self.B.shape

        hye_lengths = self.B.sum(axis=0)
        hye_counter = dict(Counter(hye_lengths))
        self.hye_count = hye_counter
        self.max_hye_size = max(hye_counter.keys())

    def get_repr(self) -> TYPE_INCIDENCE:
        return self.B

    def get_binary_incidence_matrix(self) -> TYPE_INCIDENCE:
        return self.B

    def sub_hyg(
        self,
        hyperedge_idx: np.ndarray | None = None,
    ) -> IncidenceHypergraph:
        """Produce a sub-hypergraph where only the specified hyperedges are present.

        Parameters
        ----------
        hyperedge_idx: the list of the hyperedges to keep, specified by their indices.

        Returns
        -------
        The sub-hypergraph instance.
        """
        if hyperedge_idx is None:
            return self

        B = self.B[:, hyperedge_idx]

        return IncidenceHypergraph(B)

    def __iter__(self) -> Iterable[np.ndarray]:
        return incidence_matrix_to_hye(self.B)

    def __str__(self):
        return f"{self.__class__.__name__} with N={self.N}, E={self.E}"

    @classmethod
    def load_from_txt(
        cls,
        hye_file: str | Path,
        N: int | None = None,
    ) -> IncidenceHypergraph:
        """Load a IncidenceHypergraph instance from a txt file, containing the list of
        hyperedges.

        Parameters
        ----------
        hye_file: text file containing the hyperedges.
        N: number of nodes in the hypergraph.

        Returns
        -------
        An instance of IncidenceHypergraph.
        """
        with open(hye_file, "r") as file:
            hye = (map(int, line.split(" ")) for line in file.readlines())

        return cls.load_from_hye_list(hye, N)

    @classmethod
    def load_from_hye_list(
        cls, hye_list: list[Iterable[int]], N: int | None
    ) -> IncidenceHypergraph:
        hye = list(set(tuple(sorted(set(hyperedge))) for hyperedge in hye_list))
        shape = (N, len(hye)) if N else None
        B = hye_list_to_binary_incidence(hye, shape=shape)

        return IncidenceHypergraph(B)

    @staticmethod
    def _check_and_convert_incidence(
        incidence: np.ndarray | sparse.spmatrix, sort_indices: bool
    ) -> TYPE_INCIDENCE:
        incidence = TYPE_INCIDENCE(incidence)
        # When converting to other sparse types, repeated entries are summed. In such
        # case, there could be entries different from 1. Set them to 1.
        # Similarly, if a weighted matrix is provided as configurations, flatten all non-zero
        # entries to 1.
        if not np.all(incidence.data == 1):
            warnings.warn(
                "The configurations matrix contains elements different from 0 and 1. "
                "All non-zero elements will be converted to 1."
            )
        incidence = incidence > 0

        if not np.all(incidence.data == 1):
            raise ValueError("The incidence matrix can only contain 1 and 0 values.")

        if sort_indices:
            incidence.sort_indices()

        return incidence
