from __future__ import annotations

from typing import Iterable

import numpy as np

from src.model.kappa import compute_log_kappa
from src.model.numerical import hyperedge_pi, log_binomial_coefficient


def explicit_sampling(
    p: np.ndarray,
    max_hye_size: int,
    node_assignments: np.ndarray,
    allow_repeated: bool = True,
    seed: int | None = None,
) -> list[tuple[int]]:
    """Sample a hypergraph from the generative model described in

    "Message Passing on Hypergraphs: Detectability, Phase Transitions, and Higher-Order
    Information", Ruggeri et al.

    Parameters
    ----------
    p: symmetric affinity matrix of shape (K, K), where K is the number of communities.
    max_hye_size: maximum hyperedge size allowed.
    node_assignments: array of length N, where N is the number of nodes, containing the
        node assignments to communities. Every entry needs to be between 0 and K-1
        (included), where K is the number of communities.
    allow_repeated: whether to allow for repeated hyperedges in the samples.
        In sparse regimes, repeated hyperedges have low probabilities, hence the
        approximation will yield faster sampling without significant compromises to the
        samples.
    seed: random seed.

    Returns
    -------
    The sampled hypergraph as a list of sorted tuples.
    """
    K = np.max(node_assignments) + 1
    N = len(node_assignments)
    comm_nodes = {i: np.arange(N)[node_assignments == i] for i in range(K)}
    comm_counts = [len(comm_nodes[i]) for i in range(K)]
    rng = np.random.default_rng(seed=seed)
    sampled_hypergraph = []

    # Sample binary interactions (i.e. edges between two nodes) directly one by one.
    sampled_hypergraph += _sample_binary_interactions(p, node_assignments, rng)

    log10 = np.log(10)
    log20 = np.log(20)
    for hye_size in range(3, max_hye_size + 1):
        log_kappa = compute_log_kappa(hye_size, N)
        for count in _community_count_combinations(hye_size, comm_counts):
            pi = hyperedge_pi(count, p)

            # In this case the hyperedge cannot exist.
            if pi == 0:
                continue

            # Compute the number of hyperedges satisfying the given count.
            log_n = _log_n_sharp(comm_counts, count)

            bernoulli_log_prob = np.log(pi) - log_kappa
            log_mean = bernoulli_log_prob + log_n
            log_var = log_mean + np.log(-np.expm1(bernoulli_log_prob))

            if log_n > log20 and log_var > log10:
                # Normal approximation of the binomial for n big enough and large
                # variance.
                num_hye = rng.normal(np.exp(log_mean), np.exp(0.5 * log_var))
                num_hye = np.clip(num_hye, a_min=0, a_max=None)
                num_hye = int(np.round(num_hye))
            elif (log_n > log20 and log_mean < -log10) or (
                log_n > 2 * log10 and log_mean < log10
            ):
                # Poisson approximation of the binomial for n big enough and small
                # variance.
                num_hye = rng.poisson(np.exp(log_mean))
            else:
                num_hye = rng.binomial(n=np.exp(log_n), p=np.exp(bernoulli_log_prob))

            # Sample hyperedges satisfying the count.
            if allow_repeated:
                sampled_hypergraph.extend(
                    _sample_hye_from_count(comm_nodes, count, rng)
                    for _ in range(num_hye)
                )
            else:
                sampled_hyes = set()
                while len(sampled_hyes) < num_hye:
                    sampled_hyes.add(_sample_hye_from_count(comm_nodes, count, rng))
                sampled_hypergraph.extend(sampled_hyes)

    return sampled_hypergraph


def _sample_binary_interactions(
    p: np.ndarray, node_assignments: np.ndarray, rng: np.random.Generator | None
) -> list[tuple[int]]:
    """Sample all the possible binary interactions between two nodes from their
    Bernoulli distribution.
    In sampling, it is assumed that :math::`\kappa_2=1`.
    Therefore, the Bernoulli probabilities are simply given by the affinity matrix p.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(node_assignments)
    x, y = np.meshgrid(node_assignments, node_assignments)
    bernoulli_probs = p[x, y]
    assert bernoulli_probs.shape == (N, N)

    edges = rng.binomial(n=1, p=bernoulli_probs)
    edges = np.triu(edges, 1)

    return [(int(a), int(b)) for a, b in zip(*edges.nonzero())]


def _community_count_combinations(
    n_nodes: int, comm_counts: list[int]
) -> Iterable[list[int]]:
    r"""Generate all possible community count vectors :math::`\#`.

    Parameters
    ----------
    n_nodes: number of nodes in the hyperedges.
    comm_counts: list of community counts.
        The entry i of the list specifies the total number of nodes in community i in
        the full hypergraph.

    Yields
    -------
    All the possible vectors of community counts :math::`\#`.
    """
    K = len(comm_counts)

    yield from (
        counts
        for counts in _community_count_combinations_recursive(n_nodes, comm_counts)
        if len(counts) == K
    )


def _community_count_combinations_recursive(
    n_nodes: int, comm_counts: list[int], current_K: int = 0
) -> Iterable[list[int]]:
    """Recursive helper function for `_community_count_combinations`."""
    if n_nodes == 0:
        yield [0] * (len(comm_counts) - current_K)

    elif current_K == len(comm_counts) - 1:
        if n_nodes > comm_counts[current_K]:
            yield []
        else:
            yield [n_nodes]

    else:
        for c in range(min(n_nodes, comm_counts[current_K]) + 1):
            yield from (
                [c] + other_counts
                for other_counts in _community_count_combinations_recursive(
                    n_nodes - c, comm_counts, current_K + 1
                )
            )


def _log_n_sharp(comm_counts: list[int], hye_comm_counts: list[int]) -> float:
    r"""compute the logarithm of the :math::`N_{\#}` factor.

    Parameters
    ----------
    comm_counts: the number of nodes in every community of the hypergraph, as a list of
        length K, where K is the number of communities.
    hye_comm_counts: the number of nodes in the hyperedge contained in every community,
        as a list of length K.

    Returns
    -------
    The value of :math::`N_{\#}`.
    """
    if len(comm_counts) != len(hye_comm_counts):
        raise ValueError("The inputs have different lengths.")
    return sum(
        log_binomial_coefficient(a, b) for a, b in zip(comm_counts, hye_comm_counts)
    )


def _sample_hye_from_count(
    comm_nodes: dict[int, np.ndarray],
    hye_comm_counts: list[int],
    rng: np.random.Generator | None,
) -> tuple[int]:
    """Sample a hyperedge given, for every community, the number of nodes in the
    hyperedge belonging to the community.

    Parameters
    ----------
    comm_nodes: dictionary specifying the nodes belonging to each community in the
        hypergraph.
    hye_comm_counts: list specifying at every entry i the number of nodes belonging to
        community i in the hyperedge to be sampled.
    rng: optional numpy random generator, to be utilized for sampling.

    Returns
    -------
    A hyperedge sampled satisfying hye_comm_counts.
    """
    if rng is None:
        rng = np.random.default_rng()

    hye = []
    for comm, node_count in zip(comm_nodes, hye_comm_counts):
        new_nodes = list(rng.choice(comm_nodes[comm], size=node_count, replace=False))
        hye.extend(new_nodes)

    return tuple(sorted(map(int, hye)))
