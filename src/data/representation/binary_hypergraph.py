from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class BinaryHypergraph(ABC):
    """Abstract class for the representation of hypergraphs with binary hyperedges."""

    N: int  # Number of nodes.
    E: int  # Number of hyperedges.
    max_hye_size: int  # Maximum size of the hyperedges in the hypergraph.
    hye_count: dict[
        int, int
    ]  # Hyperedges divided by hyperedge size, as (key, value) pairs: (size, count).

    @abstractmethod
    def get_repr(self) -> Any:
        """Return the internal representation of the hypergraph."""

    @abstractmethod
    def get_binary_incidence_matrix(self) -> Any:
        """Return the incidence matrix B with only zeros and ones."""

    @abstractmethod
    def __iter__(self) -> Iterable[Any]:
        """Create an iterable that yields the hyperedges."""

    def sub_hyg(self, *args: Any) -> BinaryHypergraph:
        """Return a sub-hypergraph representation."""
        raise NotImplementedError(f"Not implemented for instance of {self.__class__}")

    def save_to_txt(self, file_path: str | Path) -> None:
        file_path = Path(file_path)

        with open(file_path, "w") as hye_file:
            for hye, _ in self:
                hye_file.write(" ".join(map(str, hye)) + "\n")

    def load_from_txt(self, *args, **kwargs) -> Any:
        """Load the hypergraph from external sources."""
        raise NotImplementedError(f"Not implemented for instance of {self.__class__}")

    def max_hye_size_select(self, max_size: int) -> BinaryHypergraph:
        """Return a sub-hypergraph where hyperedges with size exceeding the one
        specified are discarded.

        Parameters
        ----------
        max_size: maximum hyperedge size allowed.

        Returns
        -------
        The sub-hypergraph where the hyperedges bigger than max_size are discarded.
        """
        incidence = self.get_binary_incidence_matrix()
        sizes = incidence.sum(axis=0)
        hye_idx = np.arange(self.E)[sizes <= max_size]
        return self.sub_hyg(hye_idx)
