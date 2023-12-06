import itertools
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
from dotenv import load_dotenv

from src.data.conversion import hye_list_to_binary_incidence
from src.data.representation.binary_hypergraph import BinaryHypergraph
from src.data.representation.incidence_hypergraph import IncidenceHypergraph
from src.model.hypergraph_block_model import HypergraphBlockModel

load_dotenv()
TEST_DATA_DIR = Path(os.environ["TEST_DATA_DIR"])


########################################################################################
# Some blockmodels.
p_vals = [
    np.array([[0.1, 0.2, 0.0], [0.2, 0.0, 0.9], [0.0, 0.9, 0.0]]),
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ),
    np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    ),
    np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 1.0, 0.0],
            [0.0, 0.0, 0.23],
        ]
    ),
]

N_vals = [2, 5, 10, 100]


def _all_models():
    for p, N in itertools.product(p_vals, N_vals):
        n = np.ones(len(p)) / len(p)
        n[-1] += 1 - n.sum()  # sum to 1, avoid numerical errors.

        assert n.sum() == 1
        yield HypergraphBlockModel(n, p, N, len(p), max_hye_size=None)


ALL_MODELS = list(_all_models())


@pytest.fixture(params=ALL_MODELS, scope="package")
def generated_blockmodel(request) -> HypergraphBlockModel:
    return request.param


########################################################################################
# Some hypergraphs.
def _load_small_hypergraphs() -> Dict[str, BinaryHypergraph]:
    hypergraphs = dict()
    for path in (TEST_DATA_DIR / "small_hypergraphs").glob("*"):
        with open(path, "r") as file:
            hye_list = [
                list(map(int, hye.strip("\n").split(" "))) for hye in file.readlines()
            ]
        binary_incidence = hye_list_to_binary_incidence(hye_list)
        hypergraphs[path.name] = IncidenceHypergraph(binary_incidence)
    return hypergraphs


SMALL_HYPERGRAPHS = _load_small_hypergraphs()


@pytest.fixture(
    params=itertools.product(SMALL_HYPERGRAPHS.values(), ALL_MODELS),
    ids=[
        name + "_" + str(i)
        for name, i in itertools.product(
            SMALL_HYPERGRAPHS.keys(), range(len(ALL_MODELS))
        )
    ],
    scope="package",
)
def hypergraph_and_model(request) -> Tuple[IncidenceHypergraph, HypergraphBlockModel]:
    hyg, model = request.param
    model.N = hyg.N
    model._init_message_passing(hyg)
    return hyg, model
