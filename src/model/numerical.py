from __future__ import annotations

import itertools

import numpy as np
from scipy import sparse, special

LN_PI = np.log(np.pi)


def sparse_reduce_lse(
    *args: sparse.csc_array | sparse.csr_array,
) -> sparse.csc_array | sparse.csr_array:
    """Perform the elementwise log-sum-exp operation on a sequence of sparse arrays.
    The arrays are assumed to have all the same pattern of non-zero entries, and to have
    sorted indices.
    """
    data = np.stack([mat.data for mat in args], axis=1)
    lse_vals = special.logsumexp(data, axis=1)

    lse_mat = args[0].copy()
    lse_mat.data = lse_vals
    return lse_mat


def log_binomial_coefficient(a: int, b: int, allow_approx: bool = True) -> float:
    """Logarithm of the binomial coefficient of a over b.

    Parameters
    ----------
    a: integer value
    b: integer value
    allow_approx: allow numerical approximation for factorials of large numbers.

    Returns
    -------
    The logarithm of the binomial coefficient of a over b.
    """
    if a == b or b == 0:
        return 0.0
    if a < b:
        raise ValueError(
            "The binomial coefficient is not defined for a smaller than b."
        )

    if not allow_approx or a - b < 5:
        log_numerator = np.sum(np.log(np.arange(a - b + 1, a + 1)))
        log_denominator = log_factorial(b)
    else:
        log_numerator = approx_log_factorial(a)
        if b > 5:
            log_denominator = approx_log_factorial(a - b) + approx_log_factorial(b)
        else:
            log_denominator = approx_log_factorial(a - b) + log_factorial(b)
    return log_numerator - log_denominator


def approx_log_factorial(a: int | float) -> float:
    """Compute :math::`\log(a!)` utilizing a Ramanujan approximation, see
    https://math.stackexchange.com/questions/152342/ramanujans-approximation-to-factorial

    Parameters
    ----------
    a: positive float or integer value

    Returns
    -------
    The approximate value of :math::`log(a!)`.
    """
    if a == 0 or a == 1:
        return 0
    if a == 2:
        return np.log(a)

    m = a * (1 + 4 * a * (1 + 2 * a))
    return a * np.log(a) - a + 0.5 * (1 / 3 * np.log(1 / 30 + m) + LN_PI)


def log_factorial(a: int | float) -> float:
    """Compute :math::`log(a!)`.

    Parameters
    ----------
    a: positive float or integer value

    Returns
    -------
    The value of :math::`log(a!)`.
    """
    if a == 0:
        return 0.0

    return np.sum(np.log(np.arange(1, a + 1)))


def hyperedge_pi(hye_comm_counts: list[int], p: np.ndarray) -> float:
    r"""Compute the value of :math::`\pi_e` for a hyperedge :math::`e`.
    The value is defined as:
    .. math::
        \pi_e := \sum_{i < j \in e} \p_{t_i t_j}

    where p is the affinity matrix and :math::`t_i` the community assignment of node i.

    Parameters
    ----------
    hye_comm_counts: a list of length K, where K is the number of communities. Every
        entry a of the list contains the number of nodes in the hyperedge belonging to
        community a.
    p: symmetric affinity matrix of probabilities in [0, 1].

    Returns
    -------
    The value of :math::`\pi_e`.
    """
    prob = 0
    for a, b in itertools.combinations(range(len(hye_comm_counts)), 2):
        prob += p[a, b] * hye_comm_counts[a] * hye_comm_counts[b]

    for a, count in enumerate(hye_comm_counts):
        prob += p[a, a] * count * (count - 1) / 2

    return prob
