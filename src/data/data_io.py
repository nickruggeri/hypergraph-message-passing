from __future__ import annotations

import pickle as pkl
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from src.data.representation.incidence_hypergraph import IncidenceHypergraph

PREPROCESSED_DATA_DIR = (
    Path(__file__).absolute().parent.parent.parent / "data" / "processed_input"
)

PREPROCESSED_DATASETS = [
    "high_school",
]


def load_real_hypergraph(
    dataset: str,
    **kwargs: Any,
) -> IncidenceHypergraph:
    """Load a real-world hypergraph.

    Parameters
    ----------
    dataset: name of the hypergraph
    kwargs: keyword arguments to be passed to the hypergraph instance created.

    Returns
    -------
    The loaded hypergraph.
    """
    if dataset not in PREPROCESSED_DATASETS:
        raise ValueError(
            f"Dataset unknown: {dataset}."
            f"\nThe available datasets are: \n{PREPROCESSED_DATASETS}"
        )

    filename = PREPROCESSED_DATA_DIR / f"{dataset}.npz"
    data = np.load(filename, allow_pickle=True)
    B = data["B"]

    # When saving sparse arrays via np.savez, they are stored inside a numpy array with
    # null shape. Manage these cases for sparse incidence B.
    if not B.shape:
        B = B.reshape(1)[0]
        assert isinstance(B, sparse.spmatrix)

    return IncidenceHypergraph(B, **kwargs)


def load_data(
    real_dataset: str = "",
    hye_file: str = "",
    pickle_file: str = "",
    N: int | None = None,
) -> IncidenceHypergraph:
    """Load a hypergraph dataset.
    Utility function for loading hypergraph data provided in various formats.
    Currently, three formats are supported:
    - real_dataset: a string with the name of a real dataset
    - hye_file: a text file containing the hyperedges
    - pickle_file: the path to a pickle serialized hypergraph.

    The function raises an error if more than one of the options above is given as
    configurations.

    Parameters
    ----------
    real_dataset: name of one the supported real datasets
    hye_file: .txt file containing the hyperedges in the dataset.
    pickle_file: path to a .pkl file to be loaded via the pickle package.
    N: number of nodes. Only utilized when hye_file is provided.

    Returns
    -------
    The loaded hypergraph.
    """
    # Check that the data is provided exactly in one of the possible configurations
    # formats.
    inputs = bool(real_dataset) + bool(hye_file) + bool(pickle_file)
    if inputs == 0:
        raise ValueError("No configurations hypergraph has been provided.")
    if inputs >= 2:
        raise ValueError("More than one configurations hypergraph has been provided.")

    if real_dataset:
        if real_dataset in PREPROCESSED_DATASETS:
            return load_real_hypergraph(real_dataset)
        raise ValueError("Unknown name for real_dataset:", real_dataset)

    if pickle_file:
        with open(pickle_file, "rb") as file:
            return pkl.load(file)

    if hye_file:
        if hye_file.endswith(".txt"):
            return IncidenceHypergraph.load_from_txt(hye_file, N)
        elif hye_file.endswith(".pkl"):
            with open(hye_file, "rb") as file:
                hye_list = pkl.load(file)
            return IncidenceHypergraph.load_from_hye_list(hye_list, N)
