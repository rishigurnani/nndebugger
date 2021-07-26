from torch_geometric.data import DataLoader
from torch import optim

def unit_sequence(input_dim, output_dim, n_hidden):
    '''
    Smoothly decay the number of hidden units in each layer.
    Start from 'input_dim' and end with 'output_dim'
    '''

    decrement = lambda x: 2**(x // 2 -1).bit_length()
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

def trainer(model, data_set, batch_size, learning_rate, n_epochs, device, loss_obj):
    
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimization
    model.train() # set model to train mode
    loss_history = []
    for epoch in range(n_epochs):
        per_epoch_loss = 0
        for ind, data in enumerate(data_loader): # loop through training batches
            data = data.to(device) # send data to GPU, if available
            optimizer.zero_grad() # zero the gradients
            output = model(data) # perform forward pass
            loss = loss_obj(output, data) # compute loss
            per_epoch_loss += loss.detach().cpu().numpy()
            loss.backward() # perform backward pass
            optimizer.step() # update weights
        loss_history.append(per_epoch_loss)
    
    return loss_history