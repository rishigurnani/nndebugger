from torch_geometric.data import DataLoader
from torch import optim
import numpy as np
import sklearn as sk
import time
from . import constants as k


def unit_sequence(input_dim, output_dim, n_hidden):
    """
    Smoothly decay the number of hidden units in each layer.
    Start from 'input_dim' and end with 'output_dim'
    """

    decrement = lambda x: 2 ** (x // 2 - 1).bit_length()
    sequence = [input_dim]
    for _ in range(n_hidden):
        last_num_units = sequence[-1]
        power2 = decrement(last_num_units)
        if power2 > output_dim:
            sequence.append(power2)
        else:
            sequence.append(last_num_units)
    sequence.append(output_dim)

    return sequence


def compute_regression_metrics(y, y_hat, mt):

    if mt:
        y = np.array(y)
        y_hat = np.array(y_hat)
        keep_inds = np.flatnonzero(y + 999)
        y_hat = y_hat[keep_inds]
        y = y[keep_inds]

    rmse = np.sqrt(sk.metrics.mean_squared_error(y, y_hat))
    r2 = sk.metrics.r2_score(y, y_hat)
    return rmse, r2


def trainer(model, data_set, batch_size, learning_rate, n_epochs, device, loss_obj):

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


def default_per_epoch_trainer(
    epoch, train_loader, model, optimizer, loss_fn, device, start
):
    y = []
    y_hat = []
    for _, data in enumerate(train_loader):  # loop through training batches
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data)
        loss.backward()
        optimizer.step()
        y += data.y.flatten().detach().cpu().numpy().tolist()
        y_hat += output.flatten().detach().cpu().numpy().tolist()
    rmse, r2 = compute_regression_metrics(y, y_hat, False)
    print("\n....Epoch %s" % epoch)
    print(f"......[rmse] {rmse} [r2] {r2}")
    print("......Outputs", y_hat[0 : k.DL_DBG_CMS_NSHOW])
    print("......Labels ", y[0 : k.DL_DBG_CMS_NSHOW])
    end = time.time()
    print(f"......Total time til this epoch {end-start}")

    return rmse, r2
