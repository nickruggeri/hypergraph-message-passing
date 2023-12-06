import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sem.str_to_type import bool_type, none_or_type

from src.model import sampling

if __name__ == "__main__":
    parser = ArgumentParser()
    # Input parameters.
    parser.add_argument(
        "-p",
        "--p",
        type=str,
        help=(
            "Symmetric matrix of community interaction probabilities."
            "The path is to be opened via numpy.loadtxt"
        ),
    )
    parser.add_argument(
        "-max_hye_size",
        "--max_hye_size",
        type=int,
        default=None,
        help="The maximum hyperedge size considered.",
    )
    parser.add_argument(
        "-node_assignments",
        "--node_assignments",
        type=str,
        default=None,
        help=(
            "Path to a file with node assignments to communities."
            "The path is to be opened via numpy.loadtxt"
        ),
    )
    parser.add_argument(
        "-allow_repeated",
        "--allow_repeated",
        type=bool_type,
        default=True,
        help=(
            "Allow repetition of hyperedges in sampling. In sparsity regimes these are "
            "negligible compared to the size of the total hyperedge ensemble."
        ),
    )
    parser.add_argument(
        "-seed",
        "--seed",
        type=none_or_type(int),
        default=None,
        help="Seed for random sampling hyepergraphs.",
    )
    parser.add_argument(
        "-save_dir",
        "--save_dir",
        type=none_or_type(Path),
        help="Directory where results are saved.",
    )

    args = parser.parse_args()

    p = np.loadtxt(args.p)
    node_assignments = np.loadtxt(args.node_assignments, dtype=int)

    hyg = sampling.explicit_sampling(
        p,
        args.max_hye_size,
        node_assignments,
        args.allow_repeated,
        args.seed,
    )

    # Serialization
    base_out = args.save_dir
    base_out.mkdir(parents=True, exist_ok=True)

    results_file = base_out / "results.pkl"
    with open(results_file, "wb") as results_f:
        pkl.dump(hyg, results_f, protocol=pkl.HIGHEST_PROTOCOL)

    args_file = base_out / "args.pkl"
    with open(args_file, "wb") as args_f:
        pkl.dump(vars(args), args_f, protocol=pkl.HIGHEST_PROTOCOL)
