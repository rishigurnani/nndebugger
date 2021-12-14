import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn, optim
from torch_geometric.loader import DataLoader

from nndebugger import torch_utils as utils


def featurize_smiles(smile, n_features):
    smile = smile.replace("*", "H")
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=n_features, useChirality=True
    )
    return np.array(fp)


class MyNet(nn.Module):
    # a logical architecture that will pass all cases
    def __init__(self, input_dim, output_dim, capacity):

        super().__init__()
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


class BuggyNet1(nn.Module):
    # buggy model. Can you spot the bug?
    def __init__(self, input_dim, output_dim, capacity):

        super().__init__()
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


def trainer(model, data_set, batch_size, learning_rate, n_epochs, device, loss_obj):
    # trainer without bugs!
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimization
    model.train()  # set model to train mode
    loss_history = []
    for epoch in range(n_epochs):
        per_epoch_loss = 0
        for ind, data in enumerate(data_loader):  # loop through training batches
            data = data.to(device)  # send data to GPU, if available
            optimizer.zero_grad()  # zero the gradients
            output = model(data)  # perform forward pass
            loss = loss_obj(output, data)  # compute loss
            per_epoch_loss += loss.detach().cpu().numpy()
            loss.backward()  # perform backward pass
            optimizer.step()  # update weights
        loss_history.append(per_epoch_loss)

    return loss_history


def buggy_trainer(
    model, data_set, batch_size, learning_rate, n_epochs, device, loss_obj
):
    # trainer with bugs! Can you spot the bug?
    # Spoiler! The bug is that there is no backward pass being performed!
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimization
    model.train()  # set model to train mode
    loss_history = []
    for epoch in range(n_epochs):
        per_epoch_loss = 0
        for ind, data in enumerate(data_loader):  # loop through training batches
            data = data.to(device)  # send data to GPU, if available
            optimizer.zero_grad()  # zero the gradients
            output = model(data)  # perform forward pass
            loss = loss_obj(output, data)  # compute loss
            per_epoch_loss += loss.detach().cpu().numpy()
            optimizer.step()  # update weights
        loss_history.append(per_epoch_loss)

    return loss_history


class BuggyNet2(nn.Module):
    # buggy model. Can you spot the "bug"?
    def __init__(self, input_dim, output_dim, capacity):

        super().__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = capacity
        unit_sequence = utils.unit_sequence(
            self.input_dim, self.output_dim, self.n_hidden
        )
        self.sigmoid = nn.Sigmoid()
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
                x = self.sigmoid(x)  # Spoiler! The "bug" is here.

        return x.view(
            data.num_graphs,
        )


def featurize_smiles_by_atom(smile, feature_dict, max_n_atoms, n_features):
    smile = smile.replace("*", "H")
    mol = Chem.MolFromSmiles(smile)
    features = np.zeros((max_n_atoms, n_features))
    for ind, atom in enumerate(mol.GetAtoms()):
        atom_feature = feature_dict[atom.GetSymbol()]
        features[ind, :] = atom_feature

    return features


class GraphNet1(nn.Module):
    def __init__(self, input_dim, output_dim, capacity, n_features, max_n_atoms):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = capacity
        self.n_features = n_features
        self.max_n_atoms = max_n_atoms

        self.layers = nn.ModuleList()
        unit_sequence = utils.unit_sequence(
            self.input_dim, self.output_dim, self.n_hidden
        )
        self.node_projector = nn.Linear(self.n_features, self.input_dim)
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
        x = x.view(data.num_graphs, self.max_n_atoms, self.n_features)
        x = self.node_projector(x)
        x_mean = x.mean(dim=2)
        x = x - x_mean[:, :, None]  # make use of broadcasting
        x = x.sum(dim=1)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < (self.n_hidden - 1):
                x = self.relu(x)

        return x.view(
            data.num_graphs,
        )


class BuggyGraphNet1(nn.Module):
    # this is a buggy model. Can you spot the bugs?
    def __init__(self, input_dim, output_dim, capacity, n_features, max_n_atoms):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden = capacity
        self.n_features = n_features
        self.max_n_atoms = max_n_atoms

        self.layers = nn.ModuleList()
        unit_sequence = utils.unit_sequence(
            self.input_dim, self.output_dim, self.n_hidden
        )
        self.node_projector = nn.Linear(self.n_features, self.input_dim)
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
        x = x.view(data.num_graphs, self.max_n_atoms, self.n_features)
        x = self.node_projector(x)
        x_mean = x.mean(dim=0)  # Spoiler! this is the bug.
        x = x - x_mean[None, :, :]  # make use of broadcasting
        x = x.sum(dim=1)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < (self.n_hidden - 1):
                x = self.relu(x)

        return x.view(
            data.num_graphs,
        )
