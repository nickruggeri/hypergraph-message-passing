import itertools

import numpy as np
import pytest
from scipy import sparse, special

from src.model.numerical import (
    approx_log_factorial,
    log_binomial_coefficient,
    log_factorial,
    sparse_reduce_lse,
)

########################################################################################
# Test sparse_reduce_lse

# Some arbitrary matrices created by hand.
matrix_list = [
    [
        np.array([[1, 1, 0], [-1, 0, 0], [2, 0, 2]]),
        np.array([[10, 1, 0], [-1, 0, 0], [3, 0, 1]]),
    ],
    [
        np.array([[-5, 0, 0, 0], [2, 3, 4, 5], [1.3, -5.1, 0, 1]]),
        np.array([[2, 0, 0, 0], [1, 2, -1.1, 1.3], [1.3, -5.1, 0, 1]]),
        np.array([[-1.1, 0, 0, 0], [3, 3, 5, 0.3], [-2.1, 2.0, 0, 1]]),
    ],
    [
        np.array([[10, 0, 100, 400], [-1, 0.3, 0, 1000], [0, 0, 0, 0]]),
        np.array([[100, 0, -100, 123], [-40, 10, 0, 1100], [0, 0, 0, 0]]),
        np.array([[102, 0, -97, 133], [-33, 11, 0, 900], [0, 0, 0, 0]]),
    ],
    [
        np.array([[10, 0, 100, 400], [-1, 0.3, 0, 1000], [2, 3, 4, 5]]),
        np.array([[100, 0, -100, 123], [-40, 10, 0, 1100], [-1, 2, 3, 5]]),
        np.array([[102, 0, -97, 133], [-33, 11, 0, 900], [1, 1, 1, 1]]),
        np.array([[-2.7, 0, 33, 133], [-33, 11, 0, 900], [1, 1, 1, 1]]),
        np.array([[-2.7, 0, 33, 133], [-33, 11, 0, 900], [1, 1, 1, 1]]),
    ],
]


# Some additional random matrices.
def generate_random_matrices_with_same_non_zeros(rng, shape, n, scale, sparsity):
    zero_idx = rng.random(shape) > sparsity

    matrices = [rng.random(shape) * scale for _ in range(n)]
    for mat in matrices:
        mat[zero_idx] = 0

    return matrices


rng = np.random.default_rng(seed=123)
shapes = [
    (10, 3),
    (100, 4),
    (1000, 50),
]
scales = [1, 10, 100]
n_matrices = [5, 10, 20]
sparsity_vals = [0.1, 0.5, 0.9]
matrix_list += [
    generate_random_matrices_with_same_non_zeros(rng, shape, n, scale, sparsity)
    for scale in scales
    for shape in shapes
    for n in n_matrices
    for sparsity in sparsity_vals
]


@pytest.fixture(params=[sparse.csc_matrix, sparse.csr_matrix])
def sparsity_type(request):
    return request.param


@pytest.fixture(params=range(len(matrix_list)))
def sparse_and_dense_matrices(sparsity_type, request):
    matrices = matrix_list[request.param]
    sparse_mat = [sparsity_type(mat) for mat in matrices]

    return matrices, sparse_mat, sparsity_type


@pytest.fixture
def sparse_and_dense_matrices_and_lse(sparse_and_dense_matrices):
    matrices, sparse_mat, sparsity_type = sparse_and_dense_matrices
    lse = sparse_reduce_lse(*sparse_mat)

    return matrices, sparse_mat, sparsity_type, lse


def test_reduce_sparse_lse_type(sparse_and_dense_matrices_and_lse):
    _, _, sparsity_type, lse = sparse_and_dense_matrices_and_lse
    assert isinstance(lse, sparsity_type)


def test_reduce_sparse_lse_with_dense(sparse_and_dense_matrices_and_lse):
    matrices, sparse_mat, sparsity_type, lse = sparse_and_dense_matrices_and_lse

    dense_lse = special.logsumexp(np.stack(matrices, axis=2), axis=2)
    dense_lse[matrices[0] == 0] = 0

    assert np.all(dense_lse == lse)


########################################################################################
# Test log_factorial, approx_log_factorial and log_binomial_coefficient
@pytest.mark.parametrize("a", range(100))
def test_stirling_approx_against_log_factorial(a):
    assert np.allclose(approx_log_factorial(a), log_factorial(a))


@pytest.mark.parametrize(
    "a,b", ((a, b) for a, b in itertools.product(range(10), range(10)) if a >= b)
)
def test_binomial_coefficient_without_approx_against_ground_truth(a, b):
    assert np.allclose(
        np.log(special.binom(a, b)),
        log_binomial_coefficient(a, b, allow_approx=False),
    )


@pytest.mark.parametrize(
    "a,b", ((a, b) for a, b in itertools.product(range(30), range(30)) if a >= b)
)
def test_binomial_coefficient_with_and_without_approx_coincide(a, b):
    assert np.allclose(
        log_binomial_coefficient(a, b, allow_approx=False),
        log_binomial_coefficient(a, b, allow_approx=True),
    )
