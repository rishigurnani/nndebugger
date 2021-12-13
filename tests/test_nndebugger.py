import pytest
# standard libraries
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import tensor, cuda, manual_seed, zeros, nn, optim
from torch import float as torch_float
from torch_geometric.loader import DataLoader
from torch import device as torch_device
import random
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# nndebugger functions
from bin.nndebugger import constants, loss, dl_debug
from bin.nndebugger import __version__
from .utils_test import featurize_smiles, MyNet, BuggyNet

# set seeds for reproducibility
random.seed(constants.RANDOM_SEED)
manual_seed(constants.RANDOM_SEED)
np.random.seed(constants.RANDOM_SEED)

def test_version():
    assert __version__ == "0.1.0"

@pytest.fixture
def example_data():
    # load data
    data_df = pd.read_csv('data/export.csv',index_col=0)
    # featurize data set
    n_features = 512
    n_data = len(data_df)
    feature_array = np.zeros((n_data, n_features))
    ind = 0
    for smiles in data_df.smiles.values:
        feature_array[ind,:] = featurize_smiles(smiles, n_features)
        ind += 1
    # prepare inputs for DebugSession
    # bug free processing pipeline!
    # data_set
    n_test = int(np.floor(n_data*constants.TRAIN_FRAC))
    n_train = n_data - n_test
    (X_train, X_test, label_train, 
    label_test) = train_test_split(
                                        feature_array,
                                        data_df.value.values.tolist(),
                                        test_size=n_test,
                                        shuffle=True,
                                        random_state=constants.RANDOM_SEED
                                    )

    train_X = [Data(x=tensor(X_train[ind,:], dtype=torch_float).view(1,n_features),
                    y=tensor(label_train[ind], dtype=torch_float)
                ) 
                for ind in range(n_train)]
    loss_fn = loss.st_loss()
    capacity_ls = [1,2,3]
    return {
        'correct_model_class_ls': [lambda : MyNet(n_features, 1, capacity) for capacity in
            capacity_ls
        ], # a list of models that are bug free!
        'model_type': 'mlp',
        'capacity_ls': capacity_ls,
        'data_set': train_X,
        'zero_data_set': [Data(x=zeros((1,n_features)), y=x.y) for x in train_X],
        'loss_fn': loss_fn,
        'device': torch_device('cuda' if cuda.is_available() else 'cpu'),
        'buggy_model_class_ls': [
            lambda : BuggyNet(n_features, 1, capacity) for capacity in capacity_ls
        ]
    }

def test_output_shape_pass(example_data):
    '''
    This output of ds.test_output_shape() should be True since we use a bug-free model
    '''
    ds = dl_debug.DebugSession(
        example_data['correct_model_class_ls'], 
        example_data['model_type'], 
        example_data['capacity_ls'], 
        example_data['data_set'], 
        example_data['zero_data_set'], 
        example_data['loss_fn'],
        example_data['device'], 
        do_test_output_shape=True
    )
    assert ds.test_output_shape()

def test_output_shape_fail(example_data):
    '''
    This output of ds.test_output_shape() should be False since we use a buggy model
    '''
    ds = dl_debug.DebugSession(
        example_data['buggy_model_class_ls'], 
        example_data['model_type'], 
        example_data['capacity_ls'], 
        example_data['data_set'], 
        example_data['zero_data_set'], 
        example_data['loss_fn'],
        example_data['device'], 
        do_test_output_shape=True
    )
    assert not ds.test_output_shape()