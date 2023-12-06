import itertools

import numpy as np
import pytest

from src.model.sampling import explicit_sampling

rng = np.random.default_rng(seed=345)
N_vals = [5, 10, 30]  # Number of nodes.
K_vals = [2, 5]  # Number of communities.
max_hye_size_vals = [2, 3, 10]  # Maximum hyperedge size.


def random_p_matrix(K, assortative, rng):
    """Generate a random affinity matrix with K communities."""
    if assortative:
        return np.diag(rng.random(size=K))
    else:
        p = rng.random(size=(K, K))
        return np.triu(p) + np.triu(p, 1).T


# Affinity matrices by number of communities.
p_vals = {
    K: [
        random_p_matrix(K, assortative, rng)
        for _ in range(3)
        for assortative in [True, False]
    ]
    for K in K_vals
}


def random_community_assignments(N, K, rng):
    """Randomly split N nodes into K communities."""
    return rng.integers(low=0, high=K, size=N)


p_and_assignments = (
    (p, random_community_assignments(N, len(p), rng))
    for matrix_group in p_vals.values()
    for p in matrix_group
    for N in N_vals
)


@pytest.fixture(
    params=(
        all_params
        for all_params in itertools.product(
            p_and_assignments,
            max_hye_size_vals,
            [True, False],
            [123, None],
        )
        # if N >= max_hye_size
        if len(all_params[0][1]) >= all_params[1]
    )
)
def sample_and_params(request):
    (
        (p, node_assignments),
        max_hye_size,
        allow_repeated,
        seed,
    ) = request.param

    param_dict = {
        "p": p,
        "node_assignments": node_assignments,
        "max_hye_size": max_hye_size,
        "allow_repeated": allow_repeated,
        "seed": seed,
    }

    return (
        explicit_sampling(p, max_hye_size, node_assignments, allow_repeated, seed),
        param_dict,
    )


def test_return_type(sample_and_params):
    sample, _ = sample_and_params
    assert isinstance(sample, list)
    assert all(isinstance(hye, tuple) for hye in sample)
    assert all(all(isinstance(node, int) for node in hye) for hye in sample)


def test_hyperedges_are_sorted(sample_and_params):
    sample, _ = sample_and_params
    assert all(hye == tuple(sorted(hye)) for hye in sample)


def test_max_hye_size_is_respected(sample_and_params):
    sample, params = sample_and_params
    if sample:
        assert max(len(hye) for hye in sample) <= params["max_hye_size"]


def test_no_allow_repeated_is_respected(sample_and_params):
    sample, params = sample_and_params
    if params["allow_repeated"]:
        pytest.skip()

    assert len(sample) == len(set(sample))


def test_sampled_nodes_go_from_0_to_N_minus_1(sample_and_params):
    sample, params = sample_and_params
    if not sample:
        return

    N = len(params["node_assignments"])
    all_nodes = set.union(*map(set, sample))
    assert min(all_nodes) >= 0
    assert min(all_nodes) <= N - 1
