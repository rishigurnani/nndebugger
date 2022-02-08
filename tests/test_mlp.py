# This file tests nndebugger functions using MLP models

import pytest

# standard libraries
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import tensor, cuda, manual_seed, zeros
from torch import float as torch_float
from torch import device as torch_device
import random
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# nndebugger functions
from nndebugger import constants, loss, dl_debug
from nndebugger import __version__
from .utils_test import (
    featurize_smiles,
    MyNet,
    BuggyNet1,
    BuggyNet2,
    trainer,
    buggy_trainer,
)

# set seeds for reproducibility
random.seed(constants.RANDOM_SEED)
manual_seed(constants.RANDOM_SEED)
np.random.seed(constants.RANDOM_SEED)


def test_version():
    assert __version__ == "0.1.0"


@pytest.fixture
def example_data():
    # load data
    data_df = pd.read_csv("data/export.csv", index_col=0)
    # featurize data set
    n_features = 512
    n_data = len(data_df)
    feature_array = np.zeros((n_data, n_features))
    ind = 0
    for smiles in data_df.smiles.values:
        feature_array[ind, :] = featurize_smiles(smiles, n_features)
        ind += 1
    # prepare inputs for DebugSession
    # bug free processing pipeline!
    # data_set
    n_test = int(np.floor(n_data * constants.TRAIN_FRAC))
    n_train = n_data - n_test
    (X_train, X_test, label_train, label_test) = train_test_split(
        feature_array,
        data_df.value.values.tolist(),
        test_size=n_test,
        shuffle=True,
        random_state=constants.RANDOM_SEED,
    )

    train_X = [
        Data(
            x=tensor(X_train[ind, :], dtype=torch_float).view(1, n_features),
            y=tensor(label_train[ind], dtype=torch_float),
        )
        for ind in range(n_train)
    ]
    loss_fn = loss.st_loss()
    capacity_ls = [1, 2, 3]
    return {
        "data_df": data_df,
        "correct_model_class_ls": [
            lambda: MyNet(n_features, 1, capacity) for capacity in capacity_ls
        ],  # a list of models that are bug free!
        "model_type": "mlp",
        "capacity_ls": capacity_ls,
        "data_set": train_X,
        "zero_data_set": [Data(x=zeros((1, n_features)), y=x.y) for x in train_X],
        "loss_fn": loss_fn,
        "device": torch_device("cuda" if cuda.is_available() else "cpu"),
        "BuggyNet1_class_ls": [
            lambda: BuggyNet1(n_features, 1, capacity) for capacity in capacity_ls
        ],
        "BuggyNet2_class_ls": [
            lambda: BuggyNet2(n_features, 1, capacity) for capacity in capacity_ls
        ],
    }


def test_output_shape_pass(example_data):
    """
    This output of ds.test_output_shape() should be True since we use a bug-free model
    """
    ds = dl_debug.DebugSession(
        example_data["correct_model_class_ls"],
        example_data["model_type"],
        example_data["capacity_ls"],
        example_data["data_set"],
        example_data["zero_data_set"],
        example_data["loss_fn"],
        example_data["device"],
        do_test_output_shape=True,
    )
    result, _ = ds.test_output_shape()
    assert result


def test_output_shape_fail(example_data):
    """
    This output of ds.test_output_shape() should be False since we use a buggy model
    """
    ds = dl_debug.DebugSession(
        example_data["BuggyNet1_class_ls"],
        example_data["model_type"],
        example_data["capacity_ls"],
        example_data["data_set"],
        example_data["zero_data_set"],
        example_data["loss_fn"],
        example_data["device"],
        do_test_output_shape=True,
    )
    result, _ = ds.test_output_shape()
    assert not result


def test_input_independent_baseline_pass(example_data):
    """
    The output of ds.test_input_independent_baseline() should be True
    since we are using a trainer without bugs
    """
    ds = dl_debug.DebugSession(
        example_data["correct_model_class_ls"],
        example_data["model_type"],
        example_data["capacity_ls"],
        example_data["data_set"],
        example_data["zero_data_set"],
        example_data["loss_fn"],
        example_data["device"],
        do_test_input_independent_baseline=True,
        trainer=trainer,
    )
    result, _ = ds.test_input_independent_baseline()
    assert result


def test_input_independent_baseline_fail(example_data):
    """
    The output of ds.test_input_independent_baseline() should be False
    since we are using a buggy trainer
    """
    ds = dl_debug.DebugSession(
        example_data["correct_model_class_ls"],
        example_data["model_type"],
        example_data["capacity_ls"],
        example_data["data_set"],
        example_data["zero_data_set"],
        example_data["loss_fn"],
        example_data["device"],
        do_test_input_independent_baseline=True,
        trainer=buggy_trainer,
    )
    result, _ = ds.test_input_independent_baseline()
    assert not result


def test_overfit_small_batch_pass(example_data):
    """
    The output of ds.test_overfit_small_batch() should be True since we are
    using a good model
    """
    ds = dl_debug.DebugSession(
        example_data["correct_model_class_ls"],
        example_data["model_type"],
        example_data["capacity_ls"],
        example_data["data_set"],
        example_data["zero_data_set"],
        example_data["loss_fn"],
        example_data["device"],
        do_test_overfit_small_batch=True,
        trainer=trainer,
    )
    result, _ = ds.test_overfit_small_batch()
    assert result


def test_overfit_small_batch_fail(example_data):
    """
    The output of ds.test_overfit_small_batch() should be True since we are
    using a good model
    """
    ds = dl_debug.DebugSession(
        example_data["BuggyNet2_class_ls"],
        example_data["model_type"],
        example_data["capacity_ls"],
        example_data["data_set"],
        example_data["zero_data_set"],
        example_data["loss_fn"],
        example_data["device"],
        do_test_overfit_small_batch=True,
        trainer=trainer,
    )
    result, _ = ds.test_overfit_small_batch()
    assert not result


def test_capacity_queue(example_data):
    capacity_ls = list(range(1, 20))
    ds = dl_debug.DebugSession(
        example_data["BuggyNet2_class_ls"] * 7,
        example_data["model_type"],
        capacity_ls,
        example_data["data_set"],
        example_data["zero_data_set"],
        example_data["loss_fn"],
        example_data["device"],
        do_test_overfit_small_batch=True,
        trainer=trainer,
        choose_model_epochs=3,
    )
    dummy_r2 = [
        # capacity 1 epoch results
        0.9,
        0.1,
        0.2,
        # capacity 2 epoch results
        0.1,
        0.8,
        0.2,
        # capacity 3 epoch results
        0.92,
        0.1,
        0.1,
        # capacity 4 epoch results
        0.2,
        0.1,
        0.94,
        # capacity 5 epoch results
        0.9,
        0.1,
        0.1,
        # capacity 6 epoch results
        0.9,
        0.1,
        0.1,
        # capacity 7 epoch results
        0.2,
        0.2,
        0.945,
    ]
    true_optimal_capacity = 4
    dummy_iter = iter(dummy_r2)

    def dummy_trainer(**kwargs):
        return 0.0, dummy_iter.__next__()

    result_optimal_capacity = ds.choose_model_size_by_overfit(
        per_epoch_trainer=dummy_trainer, patience=3, delta=0.01
    )
    assert result_optimal_capacity == true_optimal_capacity
