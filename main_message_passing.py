import logging
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sem.str_to_type import none_or_type

from src.data.data_io import load_data
from src.model import dynamic_updates
from src.model.hypergraph_block_model import HypergraphBlockModel

if __name__ == "__main__":
    parser = ArgumentParser()
    # Data IO.
    parser.add_argument(
        "--real_dataset",
        type=none_or_type(str),
        default=None,
        help="Name of a real dataset to be loaded.",
    )
    parser.add_argument(
        "--hye_file",
        type=none_or_type(str),
        default=None,
        help=(
            "Path to a file containing a list of hyperedges representing a "
            "hypergraph.",
        ),
    )
    parser.add_argument(
        "--pickle_file",
        type=none_or_type(str),
        default=None,
        help="Path to a file containing a pickle serialized hypergraph.",
    )
    # Data parameters.
    parser.add_argument(
        "--max_hye_size",
        type=none_or_type(int),
        default=None,
        help=(
            "The maximum hyperedge size considered. This value is used to exclude "
            "hyperedges in the configurations hypergraph, as well as a parameter of "
            "the probabilistic model to compute internal quantities."
        ),
    )
    parser.add_argument(
        "--save_dir", type=none_or_type(Path), help="Directory where results are saved."
    )
    # Model parameters.
    parser.add_argument(
        "--N",
        type=none_or_type(int),
        default=None,
        help=(
            "Number of nodes in the configurations hypergraph. Only needed (optionally)"
            " when specifying hye_file."
        ),
    )
    parser.add_argument(
        "--K",
        type=int,
        help="Number of communities in the model.",
    )
    parser.add_argument(
        "--n",
        type=none_or_type(str),
        default=None,
        help=(
            "Prior parameters for the communities of the stochastic block model. "
            "This is a path to a file to be opened via numpy.loadtxt. "
            "If not provided, the value of n is initialized at random. "
        ),
    )
    parser.add_argument(
        "--p",
        type=none_or_type(str),
        default=None,
        help=(
            "Symmetric matrix of community interaction probabilities. "
            "This is a path to a file to be opened via numpy.loadtxt "
            "If not provided, the value of p is initialized at random. "
        ),
    )
    # Model training.
    parser.add_argument(
        "--train_rounds",
        type=int,
        default=1,
        help=(
            "Train with different various random initializations and "
            "choose only the model attaining the best log-likelihood."
        ),
    )
    parser.add_argument(
        "--em_iter",
        type=int,
        default=20,
        help="Max iterations of the EM procedure.",
    )
    parser.add_argument(
        "--em_thresh",
        type=float,
        default=1.0e-5,
        help=(
            "Threshold for the parameter change during EM. The difference is computed "
            "with respect to the affinity matrix p and the community prior n."
        ),
    )
    parser.add_argument(
        "--mp_iter",
        type=int,
        default=2000,
        help="Max iterations of the message passing procedure.",
    )
    parser.add_argument(
        "--mp_thresh",
        type=float,
        default=1.0e-5,
        help=(
            "Threshold for the parameter change during message passing. "
            "The difference is computed with respect to the log-marginal values."
        ),
    )
    parser.add_argument(
        "--mp_patience",
        type=int,
        default=50,
        help=(
            "Number of consecutive steps where the change in log-marginals is below "
            "the mp_thresh before message passing is stopped."
        ),
    )
    parser.add_argument(
        "--dirichlet_init_alpha",
        type=none_or_type(float),
        default=None,
        help="Dirichlet alpha utilized for the model initialization.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.99,
        help="Dropout in the message passing updates.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help=(
            "Maximum number of parallel jobs. "
            "1 means no parallelization, -1 means all the available cores."
        ),
    )
    parser.add_argument(
        "--seed",
        type=none_or_type(int),
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--logging",
        type=str,
        default="INFO",
        help="Logging level.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    logging.getLogger().setLevel(args.logging.upper())

    hyg = load_data(
        args.real_dataset,
        args.hye_file,
        args.pickle_file,
        args.N,
    )
    if args.max_hye_size is not None:
        hyg = hyg.max_hye_size_select(args.max_hye_size)

    if args.n is not None:
        n = np.loadtxt(args.n)
    else:
        n = None
    if args.p is not None:
        p = np.loadtxt(args.p)
    else:
        p = None

    # Set maximum number of parallel jobs.
    dynamic_updates.N_JOBS = args.n_jobs

    best_model = None
    best_free_energy = float("inf")
    all_free_energy = []
    for i in range(args.train_rounds):
        model = HypergraphBlockModel(
            n=n, p=p, N=hyg.N, K=args.K, max_hye_size=hyg.max_hye_size
        )
        model.em_inference(
            hypergraph=hyg,
            em_iter=args.em_iter,
            em_thresh=args.em_thresh,
            mp_iter=args.mp_iter,
            mp_thresh=args.mp_thresh,
            mp_patience=args.mp_patience,
            seed=args.seed + i * 1024 if args.seed is not None else None,
            dirichlet_alpha=args.dirichlet_init_alpha,
            dropout=args.dropout,
        )
        free_energy = model.free_energy(hyg)
        all_free_energy.append(free_energy)
        if free_energy < best_free_energy:
            best_model = model
            best_free_energy = free_energy

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

        # Command line arguments.
        with open(args.save_dir / "args.txt", "w") as file:
            file.write("\n".join(sys.argv[1:]))

        # Inference results.
        np.savez(
            args.save_dir / "inferred_params.npz",
            log_marginals=best_model.log_marginals,
            log_hye_to_node=best_model.log_hye_to_node,
            log_node_to_hye=best_model.log_node_to_hye,
            external_field=best_model.external_field,
            best_free_energy=best_free_energy,
            all_free_energy=all_free_energy,
            p=best_model.p,
            n=best_model.n,
            n_diff=best_model.n_diff,
            c_diff=best_model.c_diff,
            log_marginal_diff=np.hstack(best_model.log_marginal_diff),
            mp_iter_per_em_iter=[len(x) for x in best_model.log_marginal_diff],
        )
