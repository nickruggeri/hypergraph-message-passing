import itertools
from collections import Counter

import numpy as np
import pytest
from scipy import special

from src.model.hypergraph_block_model import HypergraphBlockModel

# Fixtures generated_blockmodel and hypergraph_and_model available in package-level
# conftest.py

########################################################################################
# Test HypergraphBlockModel.single_hye_pi and HypergraphBlockModel.hye_pi
assignment_vals = [
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 2, 1, 2, 0, 0, 0, 0],
    [0, 1, 2, 3, 3, 1, 3, 3, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0],
]


def pi_brute_force(assignment, p):
    pi_brute_force = 0
    for t1, t2 in itertools.combinations(assignment, 2):
        addend = p[t1, t2]
        pi_brute_force += addend

    return pi_brute_force


@pytest.mark.parametrize("assignment", assignment_vals)
def test_pi_calculation_all_combinations(
    generated_blockmodel: HypergraphBlockModel, assignment: np.ndarray
) -> None:
    p = generated_blockmodel.p

    if p.shape[0] <= max(assignment):
        return pytest.skip("Incorrect configuration.")

    pi_explicit = pi_brute_force(assignment, p)

    assert np.allclose(generated_blockmodel.single_hye_pi(assignment), pi_explicit)


@pytest.fixture
def pi_interactions_and_data(hypergraph_and_model):
    hyg, model = hypergraph_and_model
    pi, interactions = model.hye_pi(hyg, return_interactions=True)
    return hyg, model, pi, interactions


def test_interactions_hye_brute_force(pi_interactions_and_data):
    hyg, model, pi, interactions = pi_interactions_and_data
    incidence = hyg.get_binary_incidence_matrix()
    assignments = model.community_assignments()

    interactions_brute = np.zeros((hyg.E, model.K, model.K))
    for hye in range(hyg.E):
        nodes = np.arange(hyg.N)[incidence.getcol(hye).todense().flatten().nonzero()[0]]
        assignment_ = assignments[nodes]
        counts = Counter(assignment_)
        for t1, t2 in itertools.combinations_with_replacement(counts, 2):
            if t1 == t2:
                interactions_brute[hye, t1, t1] = counts[t1] * (counts[t1] - 1) / 2
            else:
                interactions_brute[hye, t1, t2] = counts[t1] * counts[t2]
                interactions_brute[hye, t2, t1] = counts[t1] * counts[t2]
    assert np.all(interactions_brute == interactions)


def test_pi_hye_brute_force(pi_interactions_and_data):
    hyg, model, pi, interactions = pi_interactions_and_data

    incidence = hyg.get_binary_incidence_matrix()
    assignments = model.community_assignments()
    pi_explicit = np.zeros(hyg.E)
    for hye in range(hyg.E):
        nodes = np.arange(hyg.N)[incidence.getcol(hye).todense().flatten().nonzero()[0]]
        pi_explicit[hye] = pi_brute_force(assignments[nodes], model.p)

    pi = model.hye_pi(hyg, return_interactions=False)
    assert np.allclose(pi, pi_explicit)


########################################################################################
# Test HypergraphBlockModel.compute_log_marginals
@pytest.fixture
def log_marginals_and_data(hypergraph_and_model):
    hyg, model = hypergraph_and_model
    return hyg, model, model.compute_log_marginals()


def test_log_marginals_shape(log_marginals_and_data):
    _, model, log_marginals = log_marginals_and_data
    assert log_marginals.shape == (model.N, model.K)


def test_log_marginals_sum_to_1(log_marginals_and_data):
    _, _, log_marginals = log_marginals_and_data
    # A sum to 1 in log-space corresponds to a log-sum-exp of 0.
    assert np.allclose(special.logsumexp(log_marginals, axis=1), 0)


########################################################################################
# Test HypergraphBlockModel.community_assignments
@pytest.fixture
def assignments_and_data(hypergraph_and_model):
    hyg, model = hypergraph_and_model
    return hyg, model, model.community_assignments()


def test_community_assignment_shape(assignments_and_data):
    _, model, assignments = assignments_and_data
    assert assignments.shape == (model.N,)


def test_community_assignment_is_between_zero_and_number_of_communities(
    assignments_and_data,
):
    _, model, assignments = assignments_and_data
    assert np.all((assignments >= 0) & (assignments <= model.K - 1))


########################################################################################
# Test HypergraphBlockModel.updated_community_prior
def test_community_prior_explicit(hypergraph_and_model):
    hyg, model = hypergraph_and_model

    assignments = model.community_assignments()
    comm, count = np.unique(assignments, return_counts=True)
    new_n = np.zeros(model.K)
    for community, c in zip(comm, count):
        new_n[community] = c

    assert np.all(new_n == model.updated_community_prior())
