from __future__ import annotations

import logging
from collections import Counter
from typing import Iterable

import numpy as np
from scipy import sparse, special

from src.data.representation.incidence_hypergraph import IncidenceHypergraph
from src.model.kappa import compute_C_prime, compute_C_third
from src.model.numerical import hyperedge_pi, sparse_reduce_lse
from .dynamic_updates import (
    compute_psi_dynamic_programming,
    compute_psi_tilde_dynamic_programming,
)

# Define the type of sparse matrix that is utilized to store the messages during message
# passing. These can be different for messages from hyperedges to nodes and from nodes
# to hyperedges.
TYPE_HYE_TO_NODE: sparse.spmatrix = sparse.csc_array
TYPE_NODE_TO_HYE: sparse.spmatrix = sparse.csc_array

CLIP_MIN: float = -30
CLIP_MAX: float = -1e-15


class HypergraphBlockModel:
    """Hypergraph version of the Stochastic Block Model, introduced in

    "Message Passing on Hypergraphs: Detectability, Phase Transitions, and Higher-Order
    Information", Ruggeri et al.


    This probabilistic model for hypergraphs partitions the nodes into K hard
    communities, specified by an array of assignments t. The communities interact
    through a symmetric affinity matrix p, with shape (K, K). Together, the community
    assignments t and the affinity matrix p define the Bernoulli probability of the
    single hyperedges to be observed or not.
    """

    def __init__(
        self,
        n: np.ndarray | None,
        p: np.ndarray | None,
        N: int,
        K: int,
        max_hye_size: int | None,
    ) -> None:
        r"""Stochastic Block Model for Hypergraphs.
        This version of SBM considers, for every node i, hard community assignments
        :math::`t_i`, i.e. categorical assignments to one out of K communities.
        Together with a (K, K) affinity matrix, these two parameters define the
        likelihood for unweighted hypergraphs (i.e. hyperedges have weights in {0, 1}).
        A prior :math::`n=(n_1, \ldots, n_K)` for the community assignments can also be
        specified.

        Parameters
        ----------
        n: array of prior parameters for the communities.
            If specified, this array is used as initialization for EM inference,
            otherwise it is initialized at random.
            The array has length K equal to the number of communities, and specifies the
            categorical prior probabilities.
        p: symmetric matrix of community interaction probabilities.
            If specified, this matrix is used as initialization for EM inference,
            otherwise it is initialized at random.
            The matrix has shape (K, K), where K is the number of communities, and
            contains the inter and intra-community interaction probabilities,
            constrained to the [0, 1] interval.
        N: number of nodes.
        K: number of communities.
        max_hye_size: maximum size of the hyperedges D.
            Notice that this quantity is used to infer probabilistic quantities in the
            model, but is not checked against input hypergraphs.
        """

        # Model related attributes
        self._check_params(n, p, K, N, max_hye_size)
        self.n = n.copy() if n is not None else None
        self.p = p.copy() if p is not None else None
        self.N = N
        self.K = K
        self.max_hye_size: int = max_hye_size if max_hye_size is not None else N

        # Quantities inferred after message passing.
        # log of the messages from hyperedges to nodes. Stored as lists of sparse
        # matrices. For every hyperedge e and node i, the matrix at position a in the
        # list contains the messages from e to i, for community assignment a.
        self.log_hye_to_node: list[TYPE_HYE_TO_NODE] | None = None
        # log of the messages from nodes to hyperedges.
        # They are encoded similarly to the messages above.
        self.log_node_to_hye: list[TYPE_NODE_TO_HYE] | None = None
        # Other quantities, log-marginals and external field
        self.log_marginals: np.ndarray | None = None
        self.external_field: np.ndarray | None = None

        # Training diagnostics.
        self.training_iter: int | None = None
        self.n_diff: list[float] = []
        self.c_diff: list[float] = []
        self.log_marginal_diff: list[list[float]] = []

        # Random number generator.
        self.rng: np.random.Generator = np.random.default_rng()

    @property
    def c(self):
        """Return the rescaled affinity matrix c, defined as
        .. math::
            c = N p
        where N is the number of nodes and p the affinity matrix.
        """
        return self.p * self.N

    def em_inference(
        self,
        hypergraph: IncidenceHypergraph,
        em_iter: int = 20,
        em_thresh: float = 1e-5,
        mp_iter: int = 2000,
        mp_thresh: float = 1e-5,
        mp_patience: int = 50,
        seed: int | None = None,
        dirichlet_alpha: float | None = None,
        dropout: float = 0.99,
    ) -> None:
        """Perform Expectation Maximization (EM) inference on a hypergraph.
        The inference routine consist of alternating message passing, where the
        community assignments :math::`t_i` are inferred, and updates to the global
        parameters, i.e. the affinity matrix w and community priors n.
        If the affinity w or priors n are provided at initialization of the model, these
        are not inferred, but kept fixed.

        Parameters
        ----------
        hypergraph: hypergraph to perform inference on.
        em_iter: maximum number of EM iterations.
            One iteration consists of the message passing routine plus the global
            parameter updates.
        em_thresh: threshold for EM convergence.
            The threshold is computed over the absolute difference of the community
            priors and the affinity matrix between two consecutive EM iterations.
        mp_iter: maximum number of message passing iterations.
        mp_thresh: threshold for message passing convergence.
            The threshold is computed over the absolute difference of the log-marginals
            between two consecutive iterations.
        mp_patience: number of steps below the mp_thresh.
            After a number of consecutive iterations, specified by patience, with an
            absolute change in log-marginals below the mp_thresh, the message passing
            procedure is stopped.
        seed: random seed.
        dirichlet_alpha: parameter for the Dirichlet distribution.
            Utilized for the initialization of the messages, which are drawn from a
            uniform Dirichlet distribution with parameter alpha.
            If None, alpha is chosen automatically.
        dropout: dropout rate.
            The dropout rate it the number of randomly discarded updates in the messages
            and marginals. At every iteration of message passing, these discarded values
            are kept at the previous iteration value.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._check_hypergraph_vs_model_params(hypergraph)

        if self.n is None:
            fixed_n = False
            self._random_init_n()
            logging.info(f"Initialized n prior:\n{self.n}")
        else:
            fixed_n = True

        if self.p is None:
            fixed_p = False
            self._random_init_p()
            logging.info(f"Initialized rescaled affinity c=N*p:\n{self.c}")
        else:
            fixed_p = True

        for it in range(em_iter):
            logging.info(f"EM iteration {it}")

            # Local parameters: message passing.
            self.parallel_message_passing(
                hypergraph,
                mp_iter=mp_iter,
                mp_thresh=mp_thresh,
                patience=mp_patience,
                warm_start=True,
                seed=None,  # keep the current random number generator unaltered.
                dirichlet_alpha=dirichlet_alpha,
                dropout=dropout,
            )

            # Global parameters: EM updates.
            if not fixed_n or not fixed_p:
                logging.info("\tUpdates of priors n and affinity p...")
            if not fixed_n:
                old_n = self.n.copy()
                self.n = self.updated_community_prior()
                self.n_diff.append(np.abs(old_n - self.n).sum())
                logging.info(
                    f"\tCommunity prior:\n{self.n}"
                    "\n\tDifference from previous iteration: "
                    f"{self.n_diff[-1]}"
                )
            if not fixed_p:
                old_c = self.c.copy()
                self.p = self.updated_affinity_matrix(hypergraph)
                self.c_diff.append(np.abs(old_c - self.c).sum())
                logging.info(
                    f"\tRescaled affinity matrix c=N*p:\n{self.c}"
                    "\n\tDifference from previous iteration:"
                    f"{self.c_diff[-1]}"
                )

            self.training_iter = it + 1

            if not fixed_n or not fixed_p:
                param_diff = 0.0
                if not fixed_n:
                    param_diff += self.n_diff[-1]
                if not fixed_p:
                    param_diff += self.c_diff[-1]
                if param_diff <= em_thresh:
                    logging.info(
                        "Expectation-maximization threshold passed. "
                        "inference terminated."
                    )
                    break

    def parallel_message_passing(
        self,
        hypergraph: IncidenceHypergraph,
        mp_iter: int = 2000,
        mp_thresh: float = 1.0e-5,
        dirichlet_alpha: float | None = None,
        dropout: float = 0.99,
        patience: int = 50,
        seed: int | None = None,
        warm_start: bool = True,
    ) -> None:
        """Perform message passing inference of the node assignments.

        Parameters
        ----------
        hypergraph: a hypergraph.
        mp_iter: maximum number of message passing iterations.
        mp_thresh: threshold for message passing convergence.
            The threshold is computed over the absolute difference of the log-marginals
            between two consecutive iterations.
        dirichlet_alpha: parameter for the Dirichlet distribution.
            Utilized for the initialization of the messages, which are drawn from a
            uniform Dirichlet distribution with parameter alpha.
            If None, alpha is chosen automatically.
        dropout: dropout rate.
            The dropout rate it the number of randomly discarded updates in the messages
            and marginals. At every iteration of message passing, these discarded values
            are kept at the previous iteration value.
        patience: number of steps below the mp_thresh.
            After a number of consecutive iterations, specified by patience, with an
            absolute change in log-marginals below the mp_thresh, the message passing
            procedure is stopped.
        seed: random seed.
        warm_start: whether to re-initialize the messages and marginal beliefs.
        """
        logging.info("\tMessage passing...")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._check_hypergraph_vs_model_params(hypergraph)

        all_messages_init = (
            self.log_hye_to_node is not None
            and self.log_node_to_hye is not None
            and self.log_marginals is not None
            and self.external_field is not None
        )

        if not warm_start or not all_messages_init:
            alpha = 10.0 * self.K if dirichlet_alpha is None else dirichlet_alpha
            self._init_message_passing(hypergraph, dirichlet_alpha=alpha)
            logging.debug(
                f"\t\tInitialized hye to node:\n{self.log_hye_to_node[0].data[:5]}"
            )
            logging.debug(
                f"\t\tInitialized node to hye:\n{self.log_node_to_hye[0].data[:5]}"
            )
            logging.debug(f"\t\tInitialized marginals:\n{self.log_marginals[:5]}")
            logging.debug(f"\t\tInitialized external field:\n{self.external_field}")

        self.log_marginal_diff.append(list())
        patience_count = 0
        for i in range(mp_iter):
            old_log_marginals = self.log_marginals.copy()
            self._parallel_message_passing_step(hypergraph, dropout)
            self.log_marginal_diff[-1].append(
                np.abs(old_log_marginals - self.log_marginals).sum()
            )
            logging.info(
                f"\t\tMP step {i} - difference in log-marginals from previous iter: "
                f"{self.log_marginal_diff[-1][-1]}"
            )

            if self.log_marginal_diff[-1][-1] <= mp_thresh:
                patience_count += 1
            else:
                patience_count = 0

            if patience_count == patience:
                logging.info(
                    "\tMessage passing threshold passed. Message passing terminated."
                )
                break

    def _parallel_message_passing_step(
        self,
        hypergraph: IncidenceHypergraph,
        dropout: float = 0.99,
    ) -> None:
        """Perform one step of message passing, updating the messages from nodes to
        factors, the messages from factors to nodes, the marginal probabilities and
        external field."""
        inc = hypergraph.get_binary_incidence_matrix()

        # Update node to hye.
        new_node_to_hye = [None] * self.K
        for assignment in range(self.K):
            col_sum = self.log_hye_to_node[assignment].sum(axis=1)
            assert col_sum.shape == (self.N,)
            col_sum += np.log(self.n[assignment]) - self.external_field[assignment]
            col_sum = col_sum.reshape((self.N, 1))
            new_node_to_hye[assignment] = (
                TYPE_HYE_TO_NODE(inc * col_sum) - self.log_hye_to_node[assignment]
            )

        norm = sparse_reduce_lse(*new_node_to_hye)
        for assignment in range(self.K):
            new_node_to_hye[assignment].data -= norm.data
            new_node_to_hye[assignment].data = np.clip(
                new_node_to_hye[assignment].data, a_min=CLIP_MIN, a_max=CLIP_MAX
            )

        # TODO dropout could be made more efficient here. Do it or not?
        if dropout > 0:
            non_dropout_mask = (
                self.rng.random(len(self.log_node_to_hye[0].data)) >= dropout
            )
            for assignment in range(self.K):
                self.log_node_to_hye[assignment].data[
                    non_dropout_mask
                ] = new_node_to_hye[assignment].data[non_dropout_mask]
        else:
            for assignment in range(self.K):
                self.log_node_to_hye[assignment].data = new_node_to_hye[assignment].data

        logging.debug(f"\t\tUpdated node to hye:\n{self.log_node_to_hye[0].data[:5]}")

        # Update hye to node.
        if dropout > 0:
            non_dropout_mask = (
                self.rng.random(len(self.log_hye_to_node[0].data)) >= dropout
            )
        else:
            non_dropout_mask = None
        new_hye_to_node = [
            TYPE_HYE_TO_NODE(x)
            for x in compute_psi_dynamic_programming(
                hypergraph=hypergraph,
                model=self,
                mask=non_dropout_mask,
            )
        ]

        norm = sparse_reduce_lse(*new_hye_to_node)
        for assignment in range(self.K):
            new_hye_to_node[assignment].data -= norm.data
            new_hye_to_node[assignment].data = np.clip(
                new_hye_to_node[assignment].data, a_min=CLIP_MIN, a_max=CLIP_MAX
            )

        for assignment in range(self.K):
            self.log_hye_to_node[assignment].data[non_dropout_mask] = new_hye_to_node[
                assignment
            ].data

        logging.debug(f"\t\tUpdated hye to node:\n{self.log_hye_to_node[0].data[:5]}")

        # Update marginals.
        new_marginals = []
        for assignment in range(self.K):
            col_sum = self.log_hye_to_node[assignment].sum(axis=1)
            assert col_sum.shape == (self.N,)
            col_sum += np.log(self.n[assignment]) - self.external_field[assignment]
            new_marginals.append(col_sum)
        new_marginals = np.stack(new_marginals, axis=1)
        assert new_marginals.shape == (self.N, self.K)

        new_marginals = new_marginals - special.logsumexp(
            new_marginals, axis=1, keepdims=True
        )
        new_marginals = np.clip(new_marginals, a_min=CLIP_MIN, a_max=CLIP_MAX)

        if dropout > 0:
            non_dropout_mask = self.rng.random(self.N) >= dropout
            self.log_marginals[non_dropout_mask] = new_marginals[non_dropout_mask]
        else:
            self.log_marginals = new_marginals

        logging.debug(f"\t\tUpdated marginals:\n{self.log_marginals[:5]}")

        # Update external field.
        lse_term = special.logsumexp(
            a=self.log_marginals.reshape((self.N, self.K, 1)),
            b=self.c.reshape(1, self.K, self.K),
            axis=(0, 1),
        )
        assert lse_term.shape == (self.K,)

        C_prime = compute_C_prime(self.max_hye_size)
        self.external_field = C_prime / self.N * np.exp(lse_term)
        logging.debug(f"\t\tUpdated external field:\n{self.external_field}")

    def updated_community_prior(self) -> np.ndarray:
        """Parameter updates for the community priors n during EM inference.

        Returns
        -------
        The updated array of community priors.
        """
        assignments = self.community_assignments()
        comm, counts = np.unique(assignments, return_counts=True)

        n = np.zeros(self.K)
        n[comm] = counts / self.N
        return np.clip(n, a_min=1.0e-20, a_max=1.0)

    def updated_affinity_matrix(self, hypergraph: IncidenceHypergraph) -> np.ndarray:
        """Parameter updates for the affinity matrix p during EM inference.

        Parameters
        ----------
        hypergraph: a hypergraph.

        Returns
        -------
        The updated affinity matrix.
        """
        # Numerator.
        pi, interactions = self.hye_pi(hypergraph, return_interactions=True)
        numerator = np.tensordot(
            interactions, 1 / np.clip(pi, a_min=1.0e-20, a_max=None), axes=(0, 0)
        )
        assert numerator.shape == (self.K, self.K)

        # Denominator.
        C_prime = compute_C_prime(self.max_hye_size)
        denominator = (
            self.N * C_prime * (self.N * np.outer(self.n, self.n) - np.diag(self.n))
        )

        p = self.p * 2 * numerator / denominator
        return np.clip(p, a_min=1e-20, a_max=0.99)

    def community_assignments(self):
        marginals = self.log_marginals
        return np.argmax(marginals, axis=1)

    def compute_external_field(self) -> np.array:
        r"""Compute the approximate external field, defined as
        .. math::
            h(t_i) :=
                \frac{C'}{N}
                \sum_{j \in V} \sum_{t_j} c_{t_i t_j} q_j(t_j)
        where
        .. math::
            C' = \sum_{d=2}^D \binom{N-2}{d-2} \frac{1}{\kappa_d}

        Returns
        -------
        The external field h.
        """
        log_marginals = self.log_marginals
        c = self.c
        K = self.K
        N = self.N
        C_prime = compute_C_prime(self.max_hye_size)

        external_field = special.logsumexp(
            a=log_marginals.reshape(N, 1, K), b=c.reshape(1, K, K), axis=(0, 2)
        )
        assert external_field.shape == (K,)
        return C_prime / N * np.exp(external_field)

    def single_hye_pi(self, assignments: Iterable[int]) -> float:
        r"""Compute the hyperedge unnormalized probability.
        For a hyperedge e and community assignments t, the unnormalized probability is
        given by
        .. math::
            \pi_e := \sum_{i < j \in e} p_{t_i t_j}

        Parameters
        ----------
        assignments: community assignments.
            This array contains the community assignments :math::`t_i` (with values
            between 0 and K-1, where K is the number of communities) for all nodes i in
            the hyperedge.

        Returns
        -------
        The value of :math::`\pi_e`.
        """
        K = self.K
        hye_comm_counts = [0] * K
        counts = Counter(assignments)
        for comm, count in counts.items():
            hye_comm_counts[comm] = count

        return hyperedge_pi(hye_comm_counts, self.p)

    def hye_pi(
        self, hypergraph: IncidenceHypergraph, return_interactions: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        r"""Compute the hyperedge unnormalized probabilities for all the hyperedges in
        the hypergraph. For a hyperedge e, the unnormalized probability has form
        .. math::
             \pi_e := \sum_{i <j \in e} p_{t_i t_j}
        with p affinity matrix and :math::`t_i` community assignment of node i.

        Parameters
        ----------
        hypergraph: the input hypergraph.
        return_interactions: whether to optionally return the tensor of community
            interactions within hyperedges, defined as, for any hyperedge e and
            communities a, b:
            .. math::
                \#_{ab}^{(e)} := \sum_{i <j \in e} \delta_{t_i a} \delta_{t_j b}
            where :math::`\delta_{xy}` is the Dirac delta, equal to 1 if :math::`x=y`,
            else 0.
            The tensor :math::`\#` has shape (E, K, K), with E number of hyperedges and
            K number of communities.
        Returns
        -------
        The array of :math::`\pi_e` values. Optionally, the tensor of :math::`\#`
        values.
        """
        E = hypergraph.E
        K = self.K
        p = self.p
        incidence = hypergraph.get_binary_incidence_matrix()

        onehot_assignments = np.zeros((self.N, K))
        onehot_assignments[np.arange(self.N), self.community_assignments()] = 1

        counts = incidence.transpose() @ onehot_assignments
        assert counts.shape == (E, K)
        del onehot_assignments

        interactions = counts.reshape(E, 1, K) * counts.reshape(E, K, 1)
        interactions[:, np.arange(K), np.arange(K)] = counts * (counts - 1) / 2
        assert interactions.shape == (E, K, K)
        del counts

        pi = 0.5 * (
            np.sum(interactions * p.reshape(1, K, K), axis=(1, 2))
            + np.inner(interactions[:, np.arange(K), np.arange(K)], np.diagonal(p))
        )

        if return_interactions:
            return pi, interactions
        return pi

    def free_energy(self, hypergraph: IncidenceHypergraph) -> float:
        """Compute the free energy of a hypergraph utilizing the message passing
        cavity approximations. The free energy, often denoted as :math::`F = -log Z`,
        corresponds to the negative log-normalizing constant of the Boltzmann
        distribution. Z is also called the evidence of the probabilistic model.

        Parameters
        ----------
        hypergraph: hypergraph.

        Returns
        -------
        The log-likelihood value.
        """
        self._check_hypergraph_vs_model_params(hypergraph)
        K = self.K
        N = self.N
        external_field = self.compute_external_field()
        ones = np.ones(hypergraph.E)
        log_marginals = self.log_marginals
        hye_dims = hypergraph.get_binary_incidence_matrix().sum(axis=0)

        # Node-related addends.
        f_i = [
            x.tocsc().dot(ones) - external_field[k]
            for k, x in enumerate(
                compute_psi_dynamic_programming(hypergraph=hypergraph, model=self)
            )
        ]
        assert len(f_i) == K
        assert all(x.shape == (N,) for x in f_i)
        f_i = np.vstack(f_i).T
        assert f_i.shape == (N, K)
        f_i = special.logsumexp(a=f_i, b=self.n.reshape(1, -1), axis=1)
        f_i_sum = f_i.sum()

        # Edge-related addends.
        # First addend.
        first_addend = compute_psi_tilde_dynamic_programming(
            hypergraph=hypergraph, model=self
        )
        first_addend = ((hye_dims - 1) * first_addend).sum()

        # Second addend.
        log_marginal_sum = special.logsumexp(log_marginals, axis=0)
        cross_log_marginal_sum = log_marginal_sum.reshape(
            (1, K)
        ) + log_marginal_sum.reshape((K, 1))
        assert cross_log_marginal_sum.shape == (K, K)

        cross_log_marginals = log_marginals.reshape((N, 1, K)) + log_marginals.reshape(
            (N, K, 1)
        )
        assert cross_log_marginals.shape == (N, K, K)
        cross_log_marginals = special.logsumexp(cross_log_marginals, axis=0)

        second_addend = special.logsumexp(
            a=np.hstack([cross_log_marginal_sum, cross_log_marginals]),
            b=np.hstack([self.c, -self.c]),
        )
        second_addend = np.exp(second_addend)
        second_addend *= compute_C_third(self.max_hye_size) / (2 * N)

        f_e_sum = first_addend + second_addend

        return -f_i_sum + f_e_sum

    @staticmethod
    def _check_params(
        n: np.ndarray, p: np.ndarray, K: int, N: int, max_hye_size: int | None
    ) -> None:
        """Check the correctness of the initialization parameters."""
        # Check coherence between n and p.
        if n is not None:
            if not np.allclose(n.sum(), 1):
                raise ValueError(
                    "The prior parameters n for the community distribution do not "
                    "sum to 1."
                )
            if np.any(n < 0):
                raise ValueError(
                    "The prior parameters n for the community distribution contain "
                    "negative values."
                )
            if len(n.shape) != 1:
                raise ValueError(
                    "The array of prior parameters n is not one-dimensional."
                )
            if n.shape != (K,):
                raise ValueError(
                    "The array of prior parameters n has dimension different from the "
                    "number of communities K."
                )

        if p is not None:
            if not np.all(p == p.T):
                raise ValueError("The probability matrix p is not symmetric.")

            if np.any(p > 1) or np.any(p < 0):
                raise ValueError(
                    "The probability matrix p contains values outside "
                    "the (0, 1) interval."
                )

            if p.shape != (K, K):
                raise ValueError("The matrix p has shape different from (K, K).")

        if p is not None and n is not None:
            if not p.shape == (K, K):
                raise ValueError(
                    "The shapes of n and p do not match. They need to be respectively "
                    "(K,) and (K, K) for some integer K."
                )

        # Check coherence between N and max_hye_size.
        if max_hye_size is not None and max_hye_size < 2:
            raise ValueError("The max_hye_size cannot be lower than 2.")

        if max_hye_size is not None and max_hye_size > N:
            raise ValueError(
                "max_hye_size cannot be higher than the number of nodes N."
            )

    def _check_hypergraph_vs_model_params(
        self, hypergraph: IncidenceHypergraph
    ) -> None:
        """Check that the model parameters are coherent with an input hypergraph."""
        if hypergraph.N != self.N:
            raise ValueError(
                "The input hypergraph has a different number of nodes "
                "than the value specified for the model."
            )

        if hypergraph.max_hye_size > self.max_hye_size:
            raise ValueError(
                "The input hypergraph contains hyperedges bigger than the max_hye_size "
                "specified in the model."
            )

    def _random_init_n(self) -> None:
        """Random initialization of the community priors n."""
        self.n = self.rng.dirichlet(alpha=[100] * self.K)

    def _random_init_p(self) -> None:
        """Random initialization of the affinity matrix p."""
        K = self.K
        N = self.N

        p = np.ones((K, K)) / (10 * (K - 1))
        p += self.rng.random((K, K)) / 50
        p = np.triu(p, 1) + np.triu(p, 1).T
        np.fill_diagonal(p, 1.0 + self.rng.random(K) / 50)
        p /= N
        p = np.clip(p, a_min=1e-10, a_max=1.0)

        self.p = p

    def _init_message_passing(
        self,
        hypergraph: IncidenceHypergraph,
        dirichlet_alpha: float = 10.0,
    ) -> None:
        r"""Random initialization of the messages, marginal beliefs, and external field.
        The initialization is performed to respect the fixed-point conditions given by
        the message passing equations.

        Parameters
        ----------
        hypergraph: a hypergraph.
        dirichlet_alpha: parameter to initialize the messages and marginal beliefs.
            These are drawn from a Dirichlet distribution with a uniform parameter array
            :math::`(\alpha, \ldots, \alpha)` with length the number of communities.
        """
        incidence = hypergraph.get_binary_incidence_matrix()

        def random_prob_init():
            beliefs = [incidence.copy().astype(float) for _ in range(self.K)]
            vals = self.rng.dirichlet(
                [dirichlet_alpha] * len(beliefs), size=len(beliefs[0].data)
            )
            for i, belief in enumerate(beliefs):
                belief.data *= vals[:, i]

            return beliefs

        # Random initialization of messages from nodes to hyperedges.
        log_node_to_hye = random_prob_init()
        for belief in log_node_to_hye:
            belief.data = np.log(belief.data)
        self.log_node_to_hye = [TYPE_NODE_TO_HYE(mat) for mat in log_node_to_hye]

        # Random initialization of the marginal beliefs.
        marginals = self.rng.dirichlet([dirichlet_alpha] * self.K, size=self.N)
        assert marginals.shape == (self.N, self.K)
        self.log_marginals = np.log(marginals)

        # Compute external field from marginals.
        self.external_field = self.compute_external_field()

        # Infer hye to node as ratio of marginals and noe to hye
        log_hye_to_node = []
        for assignment in range(self.K):
            log_hye_to_node.append(
                TYPE_HYE_TO_NODE(
                    incidence * self.log_marginals[:, assignment].reshape(self.N, 1)
                )
                - self.log_node_to_hye[assignment]
            )

        normalizer = sparse_reduce_lse(*log_hye_to_node)
        for assignment in range(self.K):
            log_hye_to_node[assignment].data -= normalizer.data
        self.log_hye_to_node = log_hye_to_node
