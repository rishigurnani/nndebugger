import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn

from bin.nndebugger import torch_utils as utils


def featurize_smiles(smile, n_features):
    smile = smile.replace("*", "H")
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=n_features, useChirality=True
    )
    return np.array(fp)


# a logical architecture that will pass all cases
class MyNet(nn.Module):
    def __init__(self, input_dim, output_dim, capacity):

        super(MyNet, self).__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = capacity
        unit_sequence = utils.unit_sequence(
            self.input_dim, self.output_dim, self.n_hidden
        )
        self.relu = nn.ReLU()
        # set up hidden layers
        for ind, n_units in enumerate(unit_sequence[:-2]):
            size_out_ = unit_sequence[ind + 1]
            layer = nn.Linear(n_units, size_out_)
            self.layers.append(layer)

        # set up output layer
        size_in_ = unit_sequence[-2]
        size_out_ = unit_sequence[-1]
        layer = nn.Linear(size_in_, size_out_)
        self.layers.append(layer)

    def forward(self, data):
        x = data.x
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < (self.n_hidden - 1):
                x = self.relu(x)

        return x.view(
            data.num_graphs,
        )


# buggy model. Can you spot the bug?
class BuggyNet(nn.Module):
    def __init__(self, input_dim, output_dim, capacity):

        super(BuggyNet, self).__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = capacity
        unit_sequence = utils.unit_sequence(
            self.input_dim, self.output_dim, self.n_hidden
        )
        self.relu = nn.ReLU()
        # set up hidden layers
        for ind, n_units in enumerate(unit_sequence[:-2]):
            size_out_ = unit_sequence[ind + 1]
            layer = nn.Linear(n_units, size_out_)
            self.layers.append(layer)

        # set up output layer
        size_in_ = unit_sequence[-2]
        size_out_ = unit_sequence[-1]
        layer = nn.Linear(size_in_, size_out_)
        self.layers.append(layer)

    def forward(self, data):
        x = data.x
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < (self.n_hidden - 1):
                x = self.relu(x)

        return x  # Spoiler! The bug is here. The correct line is 'return x.view(data.num_graphs,)'
