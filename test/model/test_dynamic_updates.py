import itertools

import numpy as np
import pytest
from scipy import sparse, special

from src.model.dynamic_updates import (
    _compute_eta,
    _compute_eta_brute_force,
    _compute_psi_brute_force,
    compute_psi_dynamic_programming,
    compute_psi_tilde_dynamic_programming,
)
from src.model.hypergraph_block_model import TYPE_HYE_TO_NODE, TYPE_NODE_TO_HYE
from src.model.numerical import sparse_reduce_lse

# Fixture hypergraph_and_model available in package-level conftest.py

# For every test configuration, number of random samples of the messages inside
# HypergraphBlockModel. One test per sample is then run.
SEEDS = [234, 456, 678]
SCALES = [0.1, 1, 10, 10]
params = [
    {"sampling_seed": seed, "scale": scale}
    for seed, scale in itertools.product(SEEDS, SCALES)
]


def sparse_all_close(
    mat1: sparse.coo_array,
    mat2: sparse.coo_array,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """Equivalent of numpy.allclose, but for scipy sparse COO arrays."""
    # Check that indices are the same.
    if not np.all(mat1.row == mat2.row):
        return False
    if not np.all(mat1.col == mat2.col):
        return False

    # Check the data via numpy.allclose
    return np.allclose(mat1.data, mat2.data, rtol=rtol, atol=atol, equal_nan=equal_nan)


@pytest.fixture(params=params)
def hypergraph_and_initialized_model(hypergraph_and_model, request):
    def random_messages(K, incidence, type_, rng):
        all_messages = []
        for _ in range(K):
            log_messages = type_(incidence.copy())
            log_messages.data = np.log(
                np.abs(rng.random(len(log_messages.data))) * request.param["scale"]
            )
            all_messages.append(log_messages)

        normalizer = sparse_reduce_lse(*all_messages)
        all_messages = [message - normalizer for message in all_messages]
        assert np.allclose(sparse_reduce_lse(*all_messages).data, 0)
        return all_messages

    hypergraph, model = hypergraph_and_model

    incidence = hypergraph.get_binary_incidence_matrix()
    K = len(model.n)
    rng = np.random.default_rng(seed=request.param["sampling_seed"])

    model.log_hye_to_node = random_messages(K, incidence, TYPE_HYE_TO_NODE, rng)
    assert all(isinstance(mess, TYPE_HYE_TO_NODE) for mess in model.log_hye_to_node)
    model.log_node_to_hye = random_messages(K, incidence, TYPE_NODE_TO_HYE, rng)
    assert all(isinstance(mess, TYPE_NODE_TO_HYE) for mess in model.log_node_to_hye)

    return hypergraph, model


@pytest.fixture(
    params=[
        {"seed": 123, "dropout": 0.1},
        {"seed": 234, "dropout": 0.6},
        {"seed": 123, "dropout": 0.8},
    ]
)
def hypergraph_model_and_mask(hypergraph_and_initialized_model, request):
    seed, dropout = request.param["seed"], request.param["dropout"]
    rng = np.random.default_rng(seed)
    hypergraph, model = hypergraph_and_initialized_model
    mask = rng.random(len(hypergraph.get_binary_incidence_matrix().data)) > dropout

    return hypergraph, model, mask


def test_eta_hye_node_brute_force(hypergraph_and_initialized_model):
    hypergraph, model = hypergraph_and_initialized_model
    incidence = hypergraph.get_binary_incidence_matrix()
    p = model.p
    q = len(model.n)
    log_hye_to_node = model.log_hye_to_node

    for hye in range(hypergraph.E):
        nodes = incidence.getcol(hye).indices
        messages = np.stack(
            [log_hye_to_node[a][:, [hye]].data for a in range(q)], axis=1
        )
        assert messages.shape == (len(nodes), q)

        for i, node in enumerate(nodes):
            partial_message = np.delete(messages, i, axis=0)
            assert partial_message.shape == (len(nodes) - 1, q)

            eta_dynamic = _compute_eta(p, partial_message)
            eta_brute_force = _compute_eta_brute_force(model, p, partial_message)
            assert np.allclose(eta_dynamic, eta_brute_force)


def test_psi_brute_force(hypergraph_and_initialized_model):
    hypergraph, model = hypergraph_and_initialized_model

    psi_dynamic = compute_psi_dynamic_programming(hypergraph, model)
    psi_brute_force = _compute_psi_brute_force(hypergraph, model)

    assert all(
        sparse_all_close(dynamic, bf)
        for dynamic, bf in zip(psi_dynamic, psi_brute_force)
    )


def test_psi_eta_tilde_false_and_true_are_coherent(hypergraph_and_initialized_model):
    hypergraph, model = hypergraph_and_initialized_model
    log_node_to_hye = model.log_node_to_hye
    psi_eta = [x.tocsc() for x in compute_psi_dynamic_programming(hypergraph, model)]
    psi_eta_tilde = compute_psi_tilde_dynamic_programming(hypergraph, model)

    for node, hye in zip(*psi_eta[0].nonzero()):
        psi_eta_lse = special.logsumexp(
            [
                psi_eta[k][node, hye] + log_node_to_hye[k][node, hye]
                for k in range(model.K)
            ]
        )
        assert np.allclose(psi_eta_lse, psi_eta_tilde[hye])


def test_psi_with_mask_corresponds_to_psi_without_mask(hypergraph_model_and_mask):
    hypergraph, model, mask = hypergraph_model_and_mask

    psi_with_mask = compute_psi_dynamic_programming(hypergraph, model, mask=mask)
    psi_with_mask = [x.tocsc() for x in psi_with_mask]

    psi_without_mask = compute_psi_dynamic_programming(hypergraph, model, mask=None)
    psi_without_mask = [x.tocsc() for x in psi_without_mask]

    for mat1, mat2 in zip(psi_with_mask, psi_without_mask):
        assert np.all(mat1.indices == mat2.indices[mask])
        assert np.all(mat1.data == mat2.data[mask])
