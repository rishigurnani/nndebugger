# This file tests nndebugger functions using GNN models

import pytest
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from torch import tensor
from torch import float as torch_float

from .test_mlp import example_data
from .utils_test import featurize_smiles_by_atom, GraphNet1, BuggyGraphNet1
from nndebugger import constants, dl_debug


@pytest.fixture
def example_gnn_data(example_data):
    np.random.seed(constants.RANDOM_SEED)
    n_data = 4  # hard-coded
    projector_dim = 100  # hard-coded
    polymer_indices = example_data["data_df"].sample(n=n_data).index
    polymer_smiles = (
        example_data["data_df"].loc[polymer_indices, "smiles"].values.tolist()
    )
    feature_dict = {
        "C": np.array([1, 0, 0, 0]),
        "O": np.array([0, 1, 0, 0]),
        "N": np.array([0, 0, 1, 0]),
        "Cl": np.array([0, 0, 0, 1]),
    }
    n_features = len(feature_dict)
    max_n_atoms = max(
        [Chem.MolFromSmiles(smile).GetNumAtoms() for smile in polymer_smiles]
    )

    labels = example_data["data_df"].loc[polymer_indices, "value"].values

    dataset = [
        Data(
            x=tensor(
                featurize_smiles_by_atom(
                    polymer_smiles[ind], feature_dict, max_n_atoms, n_features
                ),
                dtype=torch_float,
            ),
            y=tensor(labels[ind], dtype=torch_float),
        )
        for ind in range(n_data)
    ]

    return {
        "GraphNet1_class_ls": [
            lambda: GraphNet1(projector_dim, 1, capacity, n_features, max_n_atoms)
            for capacity in example_data["capacity_ls"]
        ],  # a list of models that are bug free!
        "dataset": dataset,
        "BuggyGraphNet1_class_ls": [
            lambda: BuggyGraphNet1(projector_dim, 1, capacity, n_features, max_n_atoms)
            for capacity in example_data["capacity_ls"]
        ],  # a list of models that are buggy!
    }


def test_chart_dependencies_pass(example_gnn_data, example_data):
    """
    The output of ds.chart_dependencies() should be True since we are using a bug-free model
    """
    ds = dl_debug.DebugSession(
        example_gnn_data["GraphNet1_class_ls"],
        "gnn",
        example_data["capacity_ls"],
        example_gnn_data["dataset"],
        None,
        example_data["loss_fn"],
        example_data["device"],
        do_chart_dependencies=True,
    )
    result, _ = ds.chart_dependencies()

    assert result


def test_chart_dependencies_fail(example_gnn_data, example_data):
    """
    The output of ds.chart_dependencies() should be False since we are using a buggy model
    """
    ds = dl_debug.DebugSession(
        example_gnn_data["BuggyGraphNet1_class_ls"],
        "gnn",
        example_data["capacity_ls"],
        example_gnn_data["dataset"],
        None,
        example_data["loss_fn"],
        example_data["device"],
        do_chart_dependencies=True,
    )
    result, _ = ds.chart_dependencies()

    assert not result
