

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