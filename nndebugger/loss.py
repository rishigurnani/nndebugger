from torch import nn, manual_seed
import numpy as np 
import random

from . import constants

#fix random seeds
random.seed(constants.RANDOM_SEED)
manual_seed(constants.RANDOM_SEED)
np.random.seed(constants.RANDOM_SEED)

class st_loss(nn.Module):
    '''
    Mean squared error loss for single-task models
    '''
    def __init__(self):
        super(st_loss, self).__init__()
        self.mse_fn = nn.MSELoss()
    
    def forward(self, predictions, data):
        mse = self.mse_fn(predictions, data.y)

        return mse