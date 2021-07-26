import numpy as np
from torch import cuda, manual_seed, optim, tensor, stack
import random
from torch_geometric.data import DataLoader
import copy

from . import torch_utils as utils
from . import constants as k

dtype = cuda.FloatTensor
# fix random seeds
random.seed(k.RANDOM_SEED)
manual_seed(k.RANDOM_SEED)
np.random.seed(k.RANDOM_SEED)

class DebugSession:
    def __init__(self, model_type, model_class_ls, complexity_ls, data_set, zero_data_set, loss_fn, epsilon,
                 device, target_abs_mean_test=False, do_test_output_shape=False, do_test_input_independent_baseline=False,
                 do_test_overfit_small_batch=False, do_visualize_large_batch_training=False, do_chart_dependencies=False,
                 do_choose_model_size_by_overfit=False, LR=.001, BS=124, CHOOSE_MODEL_EPOCHS=1000):
        
        self.do_test_output_shape = do_test_output_shape
        self.do_test_input_independent_baseline = do_test_input_independent_baseline
        self.do_test_overfit_small_batch = do_test_overfit_small_batch
        self.do_visualize_large_batch_training = do_visualize_large_batch_training
        self.do_chart_dependencies = do_chart_dependencies
        self.do_choose_model_size_by_overfit = do_choose_model_size_by_overfit
        self.model_class_ls = model_class_ls
        self.model_type = model_type # should be 'gnn' for graph neural network or 'mlp' for multi-layer perceptron
        if self.model_type not in ['gnn', 'mlp']:
            raise ValueError("'model_type' can only be 'gnn' or 'mlp'.")
        
        self.data_set = data_set
        if type(self.data_set) != dict:
            raise TypeError("'data_set' must be a Python Dictionary")
        
        self.training_size = len(data_set['train'])
        print(f"Training data contains {self.training_size} points\n")
        self.zero_data_set = zero_data_set
        self.device = device
        self.target_abs_mean_test = target_abs_mean_test
        # if self.target_abs_mean_test is False:
        #     print('Skipping test_target_abs_mean', flush=True)
        self.LR = LR
        self.BS = BS
        self.CHOOSE_MODEL_EPOCHS = CHOOSE_MODEL_EPOCHS
        self.EPSILON = epsilon
        self.loss_fn = loss_fn
        self.selector_dim = None
        self.complexity_ls = complexity_ls
        if self.data_set['train'][0].__contains__('selector'):
            self.selector_dim = self.data_set['train'][0].selector.size()[-1]
        if self.selector_dim:
            self.mt = True
        else:
            self.mt = False

    def get_model_output(self,model,data):
        if self.selector_dim:
            output = model(data).view(data.num_graphs,self.selector_dim)
        else:
            # output = model(data).view(data.num_graphs,)
            output = model(data)
        return output  
    
    def test_target_abs_mean(self, model, target_abs_mean):
        model.train()
        if hasattr(model, 'init_bias'):
            model.init_bias(target_abs_mean)
        optimizer = optim.Adam(model.parameters(), lr=self.LR) #Adam optimization
        train_loader = DataLoader(self.data_set['train'], 
                                  batch_size=k.DL_DBG_TEST_MEAN_BS, 
                                  shuffle=True)
        model.train()
        for data in train_loader: #loop through training batches
            pass
        self.data = data.to(self.device) # assigned to self for test_output_shape
        optimizer.zero_grad()
        
        self.output = self.get_model_output(model,self.data) # assigned to self for test_output_shape
        if self.target_abs_mean_test:
            print('\nChecking that all outputs are near to the mean', flush=True)
            assert (np.max(np.abs(target_abs_mean - self.output.detach().cpu().numpy())) 
                                  / target_abs_mean) < k.DL_DBG_TEST_MEAN_EPS #the absolute deviation from the mean should be <.1
            print('Verified that all outputs are near to the mean\n', flush=True)

        loss = self.loss_fn(self.output, data)
        loss.backward()

    def test_output_shape(self):
        assert (self.output.shape == self.data.y.shape), f"The model output shape {self.output.shape} and label shape {self.data.y.shape} are not the same"
        print('\nVerified that shape of model predictions is equal to shape of labels\n', flush=True)

    def grad_check(self, model, file_name):
        try:
            if k.SAVE_GRAD_IMG:
                utils.plot_grad_flow(model.named_parameters(),filename=file_name)
            else:
                print('\nPlotting of gradients skipped')
        except:
            print('Error: None-type gradients detected', flush=True)
            raise
    
    def test_input_independent_baseline(self, model):
        print('\nChecking input-independent baseline', flush=True)
        zero_model = copy.deepcopy(model)
        train_loader = DataLoader(self.data_set['train'], batch_size=self.BS, shuffle=True)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.LR) #Adam optimization

        for epoch in range(k.DL_DBG_IIB_EPOCHS):
            real_data_loss = 0
            for ind, data in enumerate(train_loader): #loop through training batches
                data = data.to(self.device)
                optimizer.zero_grad()
                output = self.get_model_output(model,data)
                if ind == 0:
                    print('....Epoch %s predictions' %epoch, 
                          output.detach().cpu()
                          .numpy()[0:k.DL_DBG_IIB_NSHOW]
                         )
                loss = self.loss_fn(output, data)
                real_data_loss += loss.detach().cpu().numpy()
                loss.backward()
                optimizer.step()
        print('..last epoch real_data_loss', real_data_loss, flush=True) #loss for all points in 5th epoch gets printed

        zero_loader = DataLoader(self.zero_data_set, batch_size=self.BS, shuffle=True)
        zero_optimizer = optim.Adam(zero_model.parameters(), lr=self.LR) #Adam optimization
        zero_model.train()
        for epoch in range(k.DL_DBG_IIB_EPOCHS):
            zero_data_loss = 0
            for ind, data in enumerate(zero_loader): #loop through training batches
                data = data.to(self.device)
                zero_optimizer.zero_grad()
                output = self.get_model_output(zero_model,data)
                loss = self.loss_fn(output, data)
                zero_data_loss += loss.detach().cpu().numpy()
                loss.backward()
                zero_optimizer.step()
        print('..last epoch zero_data_loss', zero_data_loss, flush=True) #loss for all points in 5th epoch gets printed
        if zero_data_loss < real_data_loss:
            raise ValueError('The loss of zeroed inputs is less than the loss of \
                    real input. This may indicate that your model is not learning anything.')
        print('Input-independent baseline is verified\n', flush=True)
    
    def test_overfit_small_batch(self, model):
        print('\nChecking if a small batch can be overfit', flush=True)
        train_loader = DataLoader(self.data_set['train'][0:k.DL_DBG_OVERFIT_BS], 
                                  batch_size=k.DL_DBG_OVERFIT_BS, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=self.LR) #Adam optimization
        model.train()
        overfit = False
        for epoch in range(k.DL_DBG_OVERFIT_EPOCHS):
            if not overfit:
                for data in train_loader: #loop through training batches
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    output = self.get_model_output(model,data)
                    loss = self.loss_fn(output, data)
                    loss.backward()
                    optimizer.step()
                    print('..Epoch', epoch)
                    print('....Loss:', np.sqrt(loss.item()))
                    print( format_array_of_floats('....Outputs', output.flatten().detach().cpu().numpy()) )
                    print( format_array_of_floats('....Labels ', data.y.flatten().detach().cpu().numpy()) )
                    if np.sqrt(loss.item()) < self.EPSILON:
                        overfit = True

        if not overfit:
            raise ValueError('Error: Your model was not able to overfit a small batch of data')
        print('Verified that a small batch can be overfit\n', flush=True)

    def visualize_large_batch_training(self, model):
        print('\nStarting visualization of training on one large batch', flush=True)
        train_loader = DataLoader(self.data_set['train'], batch_size=k.DL_DBG_VIS_BS, shuffle=False)
        optimizer = optim.Adam(model.parameters(), lr=self.LR) #Adam optimization
        model.train()
        min_loss = np.inf
        n_epochs = self.CHOOSE_MODEL_EPOCHS // 2
        for epoch in range(n_epochs):
            for ind,data in enumerate(train_loader): #loop through training batches
                data = data.to(self.device)
                optimizer.zero_grad()
                output = self.get_model_output(model,data)
                loss = self.loss_fn(output, data)
                loss.backward()
                optimizer.step()
                if ind == 0:
                    print('..Epoch %s' %epoch)
                    print( format_array_of_floats('....Outputs', output.flatten().detach().cpu().numpy()) )
                    print( format_array_of_floats('....Labels ', data.y.flatten().detach().cpu().numpy()) )
                    print('....Loss:', np.sqrt(loss.item()))
                    if loss < min_loss:
                        min_loss = loss
                    print('....Best loss:', np.sqrt(min_loss.item()), flush=True)
        print('Visualization complete \n', flush=True)

    def chart_dependencies(self, model):
        print('\nBeginning to chart dependencies', flush=True)
    
        train_loader = DataLoader(self.data_set['train']
                                  [0:k.DL_DBG_CHART_NSHOW], 
                                  batch_size=k.DL_DBG_CHART_BS, 
                                  shuffle=False)
        optimizer = optim.Adam(model.parameters(), lr=self.LR) #Adam optimization
        model.train()
        i = 0
        for epoch in range(1):
            for data in train_loader: #loop through training batches
                data = data.to(self.device)
                data.x.requires_grad = True
                optimizer.zero_grad()
                output = self.get_model_output(model,data)
                loss = output[0].sum()
                loss.backward()
                print('..Epoch %s' %epoch)
                print( format_array_of_floats('....Outputs', output.flatten().detach().cpu().numpy()) )
                print( format_array_of_floats('....Labels ', data.y.flatten().detach().cpu().numpy()) )
                print('....Loss:', loss.item(), flush=True)

        if self.model_type is 'gnn':
            start_ind = self.data_set['train'][0].x.shape[0] #the number of nodes in the first connected graph
        elif self.model_type is 'mlp':
            start_ind = 1
        else:
            raise ValueError("Invalid 'model_type' selected")
        if data.x.grad[start_ind:,:].sum().item() != 0:
            raise ValueError('Data is getting passed along the batch dimension.')
        
        print('Finished charting dependencies. Data is not getting passed along the batch dimension.\n', flush=True)

    def choose_model_size_by_overfit(self):
        print('\nBeginning model size search', flush=True)

        N_TRAIN_DATA = len(self.data_set['train'])
        train_loader = DataLoader(self.data_set['train'], batch_size=self.BS, shuffle=False)

        min_best_rmse = np.inf
        max_best_r2 = -np.inf
        best_model_n = None #index of best model
        for model_n, model_class in enumerate(self.model_class_ls):
            print('\n..Training model %s \n' %model_n)
            model = model_class()
            if self.is_non_negative: # mean initialization only makes
                                     # sense if all values are positive
                if hasattr(model, 'init_bias'):
                    model.init_bias(self.target_abs_mean)
            optimizer = optim.Adam(model.parameters(), lr=self.LR) #Adam optimization
            model = model.to(self.device)
            model.train()
            min_rmse = np.inf #epoch-wise loss
            max_r2 = 0
            for epoch in range(self.CHOOSE_MODEL_EPOCHS):
                rmse = 0
                y = []
                y_hat = []
                for ind,data in enumerate(train_loader): #loop through training batches
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    output = self.get_model_output(model,data)
                    loss = self.loss_fn(output, data)
                    loss.backward()
                    optimizer.step()
                    y += data.y.flatten().cpu().numpy().tolist()
                    y_hat += output.flatten().detach().cpu().numpy().tolist()
                rmse, r2 = utils.compute_regression_metrics(y,y_hat,self.mt)
                print('\n....Epoch %s' %epoch)
                print(f'......[rmse] {rmse} [r2] {r2}')
                print( format_array_of_floats( '......Outputs', 
                       tensor(y_hat[0:k.DL_DBG_CMS_NSHOW]).flatten() ) )
                print( format_array_of_floats( '......Labels ', 
                       tensor(y[0:k.DL_DBG_CMS_NSHOW]).flatten() ) )
                if rmse < min_rmse:
                    min_rmse = rmse
                    max_r2 = r2

                print('......[best rmse] %s [best r2] %s' %(min_rmse, max_r2), flush=True)
            if k.SAVE_GRAD_IMG:
                utils.plot_grad_flow(model.named_parameters(),filename='big_model_%s_grad_check.png' %model_n)
                print('..Set of gradients plotted to big_model_%s_grad_check.png' %model_n, flush=True)
            else:
                print('..Plotting of gradients skipped')
            if min_rmse > min_best_rmse: # exit model testing loop if we did not improve 
                break
            elif max_best_r2 > k.DL_DBG_SUFFICIENT_R2: # exit model testing loop if this model did good enough
                break
            else:
                min_best_rmse = min_rmse
                max_best_r2 = max_r2
                best_model_n = model_n
        print('Finished model size search. Index of best model is %s\n' %best_model_n, flush=True)
        return self.complexity_ls[best_model_n]
    
    def is_non_negative(self):
        tensor = stack([x.y for x in self.data_set['train']])
        return (tensor >= 0).all().item()

    def main(self):
        
        min_model = self.model_class_ls[0]() #instantiate model
        min_model.to(self.device)
        self.non_negative = self.is_non_negative()
        self.target_abs_mean = stack([x.y for x in self.data_set['train']]
                                    ).abs().mean().item()
        # print('\ntarget_abs_mean %s \n' %self.target_abs_mean, flush=True)
        self.test_target_abs_mean(min_model, self.target_abs_mean)
        if self.do_test_output_shape:
            self.test_output_shape()
        # self.grad_check(min_model, file_name='first_grad_check.png')
        # print('\nSet of gradients plotted to first_grad_check.png\n', flush=True)

        min_model = self.model_class_ls[0]() #re-instantiate model
        min_model.to(self.device)
        if self.do_test_input_independent_baseline:
            self.test_input_independent_baseline(min_model)

        min_model = self.model_class_ls[0]() #re-instantiate model
        min_model.to(self.device)
        if self.is_non_negative and hasattr(min_model, 'init_bias'):
            min_model.init_bias(self.target_abs_mean)

        if self.do_test_overfit_small_batch:
            self.test_overfit_small_batch(min_model)

        min_model = self.model_class_ls[0]() #re-instantiate model
        min_model.to(self.device)
        if self.do_visualize_large_batch_training:
            self.visualize_large_batch_training(min_model)
        # self.grad_check(min_model, file_name='second_grad_check.png')
        # print('\nSet of gradients plotted to second_grad_check.png\n', flush=True)

        min_model = self.model_class_ls[0]() #re-instantiate model
        min_model.to(self.device)
        if self.do_chart_dependencies:
            self.chart_dependencies(min_model)

        if self.do_choose_model_size_by_overfit:
            best_model = self.choose_model_size_by_overfit()
        else:
            best_model = None
        print('\nDebug session complete.', flush=True)
        return best_model    

def format_array_of_floats(prefix, arr):
    num_ls = [f'{x:.{k.DL_DBG_NDECIMALS}f}'.format(x) for x in arr]
    ls = [prefix] + num_ls
    return ' '.join(ls)