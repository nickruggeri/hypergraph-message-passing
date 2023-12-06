from typing import Iterable

import numpy as np
import pytest
from scipy import sparse

from src.data.conversion import hye_list_to_binary_incidence, incidence_matrix_to_hye

hye_lists = [
    [
        [0, 1, 2],
        [0, 2],
        [0, 1],
    ],
    [
        [0, 3, 5],
        [0, 1],
        [5, 0, 2],
        [0, 1, 2, 3, 5],
        [5, 2, 3],
    ],
    [
        (0, 3, 5),
        [0, 1],
        (5, 0, 2),
        [0, 1, 2, 3, 5],
        (5, 2, 3),
    ],
    [
        [0, 1],
        [0, 1],
        [2, 1],
        [1, 2],
        [0, 3],
    ],
]


@pytest.mark.parametrize("hye_list", hye_lists)
def test_hye_list_to_binary_coo_incidence(hye_list: Iterable[int]) -> None:
    sparse_incidence = hye_list_to_binary_incidence(hye_list)

    N = max(map(max, hye_list)) + 1
    E = len(hye_list)
    dense_incidence = np.zeros((N, E))
    for j, hye in enumerate(hye_list):
        dense_incidence[hye, j] = 1

    assert np.all((sparse_incidence == dense_incidence).data)


incidence_matrices = [
    sparse.csr_matrix(
        np.array(
            [
                [0, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
    ),
    sparse.csr_matrix(
        np.array(
            [
                [1, 1, 0],
                [0, 1, 1],
                [1, 1, 0],
            ]
        )
    ),
    sparse.csr_matrix(
        np.array(
            [
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        )
    ),
]
incidence_matrices = (
    incidence_matrices
    + list(map(sparse.csc_matrix, incidence_matrices))
    + list(map(sparse.coo_matrix, incidence_matrices))
    + list(map(sparse.lil_matrix, incidence_matrices))
)


@pytest.mark.parametrize("incidence", incidence_matrices)
def test_incidence_matrix_to_hye(incidence: sparse.spmatrix) -> None:
    generated_list = list(incidence_matrix_to_hye(incidence))

    dense_incidence = incidence.todense()
    hye_list_from_dense = [
        dense_incidence[:, j].nonzero()[0] for j in range(dense_incidence.shape[1])
    ]

    assert all(
        np.all(hye1 == hye2) for hye1, hye2 in zip(generated_list, hye_list_from_dense)
    )
