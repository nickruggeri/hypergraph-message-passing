import itertools
from collections import Counter
from typing import Dict, List

import numpy as np
import pytest
from scipy import special

from src.model.sampling import (
    _community_count_combinations,
    _log_n_sharp,
    _sample_hye_from_count,
)

n_nodes_all = [2, 5, 10, 25, 50, 100]
rng = np.random.default_rng(seed=123)
hye_comm_counts_all = [
    rng.integers(low=0, high=max_val, size=q)
    for _ in range(10)
    for max_val in [5, 10]
    for q in [2, 3, 4, 5]
]
comm_counts_all = sum(
    (
        [
            hye_comm_count + rng.integers(low=0, high=high, size=len(hye_comm_count))
            for hye_comm_count in hye_comm_counts_all
        ]
        for high in [1, 5, 10]
    ),
    start=[],
)
hye_comm_counts_all = [list(x) for x in hye_comm_counts_all]
comm_counts_all = [list(x) for x in comm_counts_all]


def generate_communities(comm_counts: List[int]) -> Dict[int, np.ndarray]:
    N = sum(comm_counts)
    K = len(comm_counts)
    rng_tmp = np.random.default_rng(seed=21)

    all_nodes = np.arange(N)
    rng_tmp.shuffle(all_nodes)
    cumcount = [0] + list(np.cumsum(comm_counts))
    comm_nodes = dict()
    for comm in range(K):
        comm_nodes[comm] = all_nodes[cumcount[comm] : cumcount[comm + 1]]
    return comm_nodes


commm_nodes_all = [generate_communities(comm_counts) for comm_counts in comm_counts_all]


########################################################################################
# Test _community_count_combinations, _log_n_sharp
@pytest.mark.parametrize(
    "n_nodes, hye_comm_counts", itertools.product(n_nodes_all, hye_comm_counts_all)
)
def test_community_count_combinations_brute_force(n_nodes, hye_comm_counts):
    all_combinations = itertools.product(*(range(a + 1) for a in hye_comm_counts))
    all_combinations = [list(comb) for comb in all_combinations if n_nodes == sum(comb)]

    assert sorted(all_combinations) == sorted(
        _community_count_combinations(n_nodes, hye_comm_counts)
    )


@pytest.mark.parametrize(
    "comm_counts, hye_comm_counts",
    zip(comm_counts_all, hye_comm_counts_all * 3),
)
def test_log_n_sharp_brute_force(comm_counts, hye_comm_counts):
    brute_force = [special.binom(a, b) for a, b in zip(comm_counts, hye_comm_counts)]
    brute_force = np.sum(np.log(brute_force))

    assert np.allclose(brute_force, _log_n_sharp(comm_counts, hye_comm_counts))


########################################################################################
# Test _sample_hye_from_count
@pytest.fixture(
    params=(
        (comm_nodes, hye_comm_counts, rng)
        for comm_nodes, hye_comm_counts in zip(commm_nodes_all, hye_comm_counts_all * 3)
        for rgn in [None, np.random.default_rng(seed=34)]
    )
)
def sampled_hye_with_info(request):
    comm_nodes, hye_comm_counts, rng = request.param
    node_to_comm = {node: comm for comm in comm_nodes for node in comm_nodes[comm]}
    return (
        _sample_hye_from_count(comm_nodes, hye_comm_counts, rng),
        comm_nodes,
        hye_comm_counts,
        node_to_comm,
    )


def test_sample_hye_from_count_returns_tuples_of_integers(sampled_hye_with_info):
    sampled_hye, _, _, _ = sampled_hye_with_info
    assert isinstance(sampled_hye, tuple)
    assert all(isinstance(node, int) for node in sampled_hye)


def test_sampled_nodes_respect_the_community_counts(sampled_hye_with_info):
    sampled_hye, _, hye_comm_counts, node_to_comm = sampled_hye_with_info
    comm_counts = [0] * len(hye_comm_counts)
    for comm, count in Counter(node_to_comm[node] for node in sampled_hye).items():
        comm_counts[comm] = count
    assert comm_counts == hye_comm_counts
