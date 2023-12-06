from __future__ import annotations

import numpy as np

from src.model.numerical import log_binomial_coefficient

LOG_KAPPA_VALS: dict[int, float] = dict()
C_PRIME_VALS: dict[int, float] = dict()


def compute_log_kappa(d: int, N: int, cache=True) -> float:
    r"""Logarithm of the :math::`\kappa` normalizing constant.
    The constant is defined as
    .. math::
        \kappa_d = d (d - 1) / 2 \binom{N-2}{d-2}
    where d is the hyperedge size and N the number of nodes in the hypergraph.

    Parameters
    ----------
    d: hyperedge size.
    N: number of nodes in the hypergraph.
    cache: whether to utilize and store cached computations.
        If True, check whether this computation has already been performed with same
        N, d values. Alternatively, cache the result.

    Returns
    -------
    The logarithm of the kappa normalizing constant.
    """
    if cache and (N, d) in LOG_KAPPA_VALS:
        return LOG_KAPPA_VALS[(N, d)]

    log_kappa = float(
        np.log(d) + np.log(d - 1) - np.log(2) + log_binomial_coefficient(N - 2, d - 2)
    )
    if cache:
        LOG_KAPPA_VALS[(N, d)] = log_kappa

    return log_kappa


def compute_C_prime(max_hye_size: int) -> float:
    r"""Compute the :math::`C'` constant defined as
    .. math::
        C' := \sum_{d=2}^D \binom{N-2}{d-2} / \kappa_d
    where D is the maximum hyperedge size, N the number of nodes in the hypergraph, and
    :math::`\kappa_d` the normalizing constant.
    """
    if max_hye_size in C_PRIME_VALS:
        return C_PRIME_VALS[max_hye_size]

    hye_dims = np.arange(2, max_hye_size + 1)
    c_prime = 2 * (1 / (hye_dims * (hye_dims - 1))).sum()
    C_PRIME_VALS[max_hye_size] = c_prime
    return c_prime


def compute_C_third(max_hye_size: int) -> float:
    r"""Compute the :math::`C'` constant defined as
    .. math::
        C''' := \sum_{d=2}^D \frac{1-d}{\kappa_d} \binom{N-2}{d-2} /
    where D is the maximum hyperedge size, N the number of nodes in the hypergraph, and
    :math::`\kappa_d` the normalizing constant.
    """
    hye_dims = np.arange(2, max_hye_size + 1)
    return -2 * (1 / hye_dims).sum()
