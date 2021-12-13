import copy
import random
import time 
import numpy as np
from torch import cuda, manual_seed, optim, tensor, stack, backends, autograd, float16
from torch_geometric.loader import DataLoader
backends.cudnn.benchmark = True

from . import torch_utils as utils
from . import constants as k

dtype = cuda.FloatTensor
# fix random seeds
random.seed(k.RANDOM_SEED)
manual_seed(k.RANDOM_SEED)
np.random.seed(k.RANDOM_SEED)


# I read that these three lines below speed up training. However, keeping
# them set makes it difficult to do profiling. 
autograd.set_detect_anomaly(False)
autograd.profiler.profile(False)
autograd.profiler.emit_nvtx(False)
np.set_printoptions(precision=3)

class DebugSession:
    """
    A class to run Debugging of neural nets
    """

    def __init__(
        self,
        model_class_ls,
        model_type,
        capacity_ls,
        data_set,
        zero_data_set,
        loss_fn,
        device,
        target_abs_mean_test=False,
        do_all_tests=False,
        do_test_output_shape=False,
        do_test_input_independent_baseline=False,
        do_test_overfit_small_batch=False,
        do_visualize_large_batch_training=False,
        do_chart_dependencies=False,
        do_choose_model_size_by_overfit=False,
        r_learn=0.001,
        batch_size=124,
        choose_model_epochs=k.DL_DBG_MAX_EPOCHS,
        trainer=utils.trainer,
    ):
        """
        Keyword Arguments:
        model_class_ls - A list of functions that return a PyTorch nn.Module instance
        capacity_ls - A list of the capacity of each function in model_class_ls
        model_type - 'mlp' if the multi-layer perceptrons are contained
        in model_class_ls or 'gnn' if graph neural networks are contained
        in model_class_ls
        data_set - A list of Data instances. Each
        Data instance should at least have an attribute 'x' which points
        to features and an attribute 'y' which points to the label
        loss_fn - An nn.Module instance. The instance should have
        a forward method that takes in two arguments, 'predictions' and
        'data'. The forward method should return a scalar.
        device - A PyTorch device.
        r_learn - A scalar indicating the learning rate that will be used.
        batch_size - An integer indicating the batch size that will be used.
        choose_model_epochs - An integer indicating the number of epochs
        that will be used in 'choose_model_size_by_overfit'
        trainer - A function that optimizes weights over several epochs
        and returns a record of the loss over every epoch. See 'trainer'
        in torch_utils.py for example inputs & outputs.
        """
        self.do_test_output_shape = do_test_output_shape
        self.do_test_input_independent_baseline = do_test_input_independent_baseline
        self.do_test_overfit_small_batch = do_test_overfit_small_batch
        self.do_visualize_large_batch_training = do_visualize_large_batch_training
        self.do_chart_dependencies = do_chart_dependencies
        self.do_choose_model_size_by_overfit = do_choose_model_size_by_overfit
        if do_all_tests:
            self.do_test_output_shape = True
            self.do_test_input_independent_baseline = True
            self.do_test_overfit_small_batch = True
            self.do_visualize_large_batch_training = True
            self.do_chart_dependencies = True
            self.do_choose_model_size_by_overfit = True
        self.model_class_ls = model_class_ls
        self.model_type = model_type  # should be 'gnn' for graph neural network or 'mlp' for multi-layer perceptron
        self.trainer = trainer
        if self.model_type not in ["gnn", "mlp"]:
            raise ValueError("'model_type' can only be 'gnn' or 'mlp'.")

        self.data_set = data_set
        if not isinstance(self.data_set, list):
            raise TypeError("'data_set' must be a Python list")

        self.training_size = len(data_set)
        print(f"Training data contains {self.training_size} points\n")
        self.zero_data_set = zero_data_set
        self.device = device
        self.target_abs_mean_test = target_abs_mean_test
        if self.target_abs_mean_test is False:
            print("Skipping test_target_abs_mean", flush=True)
        self.r_learn = r_learn
        self.batch_size = batch_size
        self.choose_model_epochs = choose_model_epochs
        self.loss_fn = loss_fn
        self.selector_dim = None
        self.capacity_ls = capacity_ls
        if self.data_set[0].__contains__("selector"):
            self.selector_dim = self.data_set[0].selector.size()[-1]
        self.multi_task = bool(self.selector_dim)

    def test_target_abs_mean(self, target_abs_mean):
        """
        Test if all model outputs are near to the mean
        """
        model, optimizer = self.initialize_training(self.model_class_ls[0])
        model = model.to(self.device)
        model.train()
        if hasattr(model, "init_bias"):
            model.init_bias(target_abs_mean)
        train_loader = DataLoader(
            self.data_set,
            batch_size=k.DL_DBG_TEST_MEAN_BS,
            shuffle=True,
            drop_last=False,
        )
        model.train()
        for data in train_loader:  # loop through training batches
            pass
        data = data.to(self.device)  # assigned to self for test_output_shape
        optimizer.zero_grad()

        output = model(data)
        if self.target_abs_mean_test:
            print("\nChecking that all outputs are near to the mean", flush=True)
            assert (
                np.max(np.abs(target_abs_mean - output.detach().cpu().numpy()))
                / target_abs_mean
            ) < k.DL_DBG_TEST_MEAN_EPS  # the absolute deviation from the mean should be <.1
            print("Verified that all outputs are near to the mean\n", flush=True)

        loss = self.loss_fn(output, data)
        loss.backward()

    def test_output_shape(self):
        """
        The shape of the model output should match the shape of the labels.
        """
        model, optimizer = self.initialize_training(self.model_class_ls[0])
        model = model.to(self.device)
        model.train()
        train_loader = DataLoader(
            self.data_set,
            batch_size=k.DL_DBG_TEST_MEAN_BS,
            shuffle=True,
            drop_last=False,
        )
        model.train()
        for data in train_loader:  # loop through training batches
            pass
        data = data.to(self.device)  # assigned to self for test_output_shape
        output = model(data)

        return (output.shape == data.y.shape, f"The model output shape {output.shape} and label shape {data.y.shape} are not the same")

    def test_input_independent_baseline(self):
        """
        The loss of the model should be lower when real features are
        passed in than when zeroed features are passed in.
        """
        print("\nChecking input-independent baseline", flush=True)
        model, optimizer = self.initialize_training(self.model_class_ls[0])
        model = model.to(self.device)
        zero_model = copy.deepcopy(model)  # deepcopy the model before training
        loss_history = self.trainer(
            model,
            self.data_set,
            self.batch_size,
            self.r_learn,
            k.DL_DBG_IIB_EPOCHS,
            self.device,
            self.loss_fn,
        )
        print(
            "..last epoch real_data_loss", 
            loss_history[-1], 
            flush=True
        )  # loss for all points in final epoch gets printed

        zero_loss_history = self.trainer(
            zero_model,
            self.zero_data_set,
            self.batch_size,
            self.r_learn,
            k.DL_DBG_IIB_EPOCHS,
            self.device,
            self.loss_fn,
        )
        print(
            "..last epoch zero_data_loss", 
            zero_loss_history[-1], 
            flush=True
        )  # loss for all points in 5th epoch gets printed
        message = '''The loss of zeroed inputs is nearly the same as the loss of
                    real inputs. This may indicate that your model is not learning anything
                    during training. Check your trainer function and your model architecture.'''
        return (loss_history[-1] / zero_loss_history[-1]) < k.DL_DBG_IIB_THRESHOLD, message

    def test_overfit_small_batch(self):
        """
        If you hope to learn a good map on your whole data set using
        model archicture A, then A should have enough capacity to
        completely overfit a small batch of the data set.
        """
        print("\nChecking if a small batch can be overfit", flush=True)
        model, optimizer = self.initialize_training(self.model_class_ls[0])
        model = model.to(self.device)
        train_loader = DataLoader(
            self.data_set[0 : k.DL_DBG_OVERFIT_BS],
            batch_size=k.DL_DBG_OVERFIT_BS,
            shuffle=True,
            drop_last=True,          
        )
        model.train()
        overfit = False
        for epoch in range(k.DL_DBG_OVERFIT_SMALL_EPOCHS):
            if not overfit:
                for data in train_loader:  # loop through training batches
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = self.loss_fn(output, data)
                    loss.backward()
                    optimizer.step()
                    y = data.y.flatten().detach().cpu().numpy()
                    y_hat = output.flatten().detach().cpu().numpy()
                    _, r2 = utils.compute_regression_metrics(y, y_hat, self.multi_task)
                    print("..Epoch", epoch)
                    print("....Outputs", y_hat)
                    print("....Labels ", y)
                    print(
                        "....Loss:", 
                        np.sqrt([loss.item()])
                    )
                    print(
                        "....R2:", r2
                    )
                    if r2 > k.DL_DBG_SUFFICIENT_R2_SMALL_BATCH:
                        overfit = True

        return overfit, f'''Error: Your model was not able to overfit a small batch 
            of data. The maximum R2 over {k.DL_DBG_OVERFIT_EPOCHS} epochs was not greater than {k.DL_DBG_SUFFICIENT_R2_SMALL_BATCH}'''

    def visualize_large_batch_training(self):
        """
        Visualize how training proceeds on a large batch of data
        """
        print("\nStarting visualization of training on one large batch", flush=True)
        model, optimizer = self.initialize_training(self.model_class_ls[0])
        model = model.to(self.device)
        train_loader = DataLoader(
            self.data_set, batch_size=k.DL_DBG_VIS_BS, shuffle=False, drop_last=True,
        )
        model.train()
        min_loss = np.inf
        n_epochs = self.choose_model_epochs // 2
        for epoch in range(n_epochs):
            for ind, data in enumerate(train_loader):  # loop through training batches
                data = data.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = self.loss_fn(output, data)
                loss.backward()
                optimizer.step()
                if ind == 0:
                    print("..Epoch %s" % epoch)
                    print(
                        "....Outputs", output.flatten().detach().cpu().numpy()
                    )
                    print(
                        "....Labels ", data.y.flatten().detach().cpu().numpy()
                    )
                    print(
                        "....Loss:", 
                        np.sqrt([loss.item()])
                    )
                    if loss < min_loss:
                        min_loss = loss
                    print(
                        "....Best loss:", 
                        np.sqrt([min_loss.item()]), 
                        flush=True
                    )
        print("Visualization complete \n", flush=True)

    def chart_dependencies(self):
        """
        Check that the forward method does not mix information from separate instances.
        """
        print("\nBeginning to chart dependencies", flush=True)
        model, optimizer = self.initialize_training(self.model_class_ls[0])
        model = model.to(self.device)
        train_loader = DataLoader(
            self.data_set[0 : k.DL_DBG_CHART_NSHOW],
            batch_size=k.DL_DBG_CHART_BS,
            shuffle=False,
            drop_last=False, # if we set drop_last equal to True then it is
            # possible for train_loader to be empty. That would mess up
            # this test.
        )
        model.train()
        for epoch in range(1):
            for data in train_loader:  # loop through training batches
                data = data.to(self.device)
                data.x.requires_grad = True
                optimizer.zero_grad()
                output = model(data)
                loss = output[0].sum()
                loss.backward()
                print("..Epoch %s" % epoch)
                print(
                    "....Outputs", output.flatten().detach().cpu().numpy()
                )
                print(
                    "....Labels ", data.y.flatten().detach().cpu().numpy()
                )
                print(
                    "....Loss:", 
                    loss.item(), 
                    flush=True
                )

        if self.model_type == "gnn":
            start_ind = self.data_set[0].x.shape[
                0
            ]  # the number of nodes in the first connected graph
        elif self.model_type == "mlp":
            start_ind = 1
        else:
            raise ValueError("Invalid 'model_type' selected")
        message = "Data is getting mixed between instances in the same batch."
        
        return (data.x.grad[start_ind:, :].sum().item() == 0, message)

    def initialize_training(self, model_class):
        """
        Initialize a model and optimizer for training using just the model's class
        """
        model = model_class()
        optimizer = optim.Adam(model.parameters(), lr=self.r_learn)  # Adam optimization
        model = model.to(self.device)
        model.train()

        return model, optimizer

    def choose_model_size_by_overfit(self):
        """
        Return the smallest capacity capable of over-fitting the data or
        the capacity with the highest R2.
        """
        print("\nBeginning model size search", flush=True)

        train_loader = DataLoader(
            self.data_set, batch_size=self.batch_size, shuffle=True, 
            drop_last=True,
        )

        min_best_rmse = np.inf
        max_best_r2 = -np.inf
        best_model_n = None  # index of best model
        overfit = False  # overfit is set to True inside the loop if we overfit the data
        for model_n, model_class in enumerate(self.model_class_ls):
            if overfit:
                break
            print("\n..Training model %s \n" % model_n)
            model, optimizer = self.initialize_training(model_class)
            min_rmse = np.inf  # epoch-wise loss
            max_r2 = -np.inf
            start = time.time()
            for epoch in range(self.choose_model_epochs):
                if overfit:
                    break
                y = []
                y_hat = []
                for _, data in enumerate(train_loader):  # loop through training batches
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = self.loss_fn(output, data)
                    loss.backward()
                    optimizer.step()
                    y += data.y.flatten().detach().cpu().numpy().tolist()
                    y_hat += output.flatten().detach().cpu().numpy().tolist()
                rmse, r2 = utils.compute_regression_metrics(y, y_hat, self.multi_task)
                print("\n....Epoch %s" % epoch)
                print(f"......[rmse] {rmse} [r2] {r2}")
                print(
                    "......Outputs", y_hat[0 : k.DL_DBG_CMS_NSHOW]
                )
                print(
                    "......Labels ", y[0 : k.DL_DBG_CMS_NSHOW]
                )
                end = time.time()
                print(f"......Total time til this epoch {end-start}")
                if rmse < min_rmse:
                    # OK, if we reached here then that means the
                    # updates seen during this epoch yield the 
                    # best predictions.
                    min_rmse = rmse
                    max_r2 = r2
                if (
                    max_r2 > k.DL_DBG_SUFFICIENT_R2_ALL_DATA
                ):  # exit model testing loop if this model did good enough
                    best_model_n = model_n
                    print(
                        "Data was overfit with capacity %s"
                        % self.capacity_ls[best_model_n],
                        flush=True,
                    )
                    overfit = True
                # if the r2 is too negative then we should re-start training to avoid NaN.
                if r2 < -(10 ** 8):
                    model, optimizer = self.initialize_training(model_class)

                print(
                    "......[best rmse] %s [best r2] %s" % (
                        min_rmse, 
                        max_r2
                    ), flush=True
                )

                # end of one epoch
            if overfit:
                break
            # OK, we did not do good enough to overfit the data. But,
            # if this capacity was the best we have seen yet, take note.
            if min_rmse < min_best_rmse:
                min_best_rmse, max_best_r2, best_model_n = min_rmse, max_r2, model_n
            # end of loop over all epochs

        print(
            "Finished model size search. The optimal capacity is %s\n"
            % self.capacity_ls[best_model_n],
            flush=True,
        )
        
        return self.capacity_ls[best_model_n]

    def is_non_negative(self):
        """
        Check if self.data_set is non-negative
        """
        stacked_tensor = stack([x.y for x in self.data_set])
        return (stacked_tensor >= 0).all().item()

    def main(self):
        """
        Call this method to perform the tests
        """
        self.non_negative = self.is_non_negative()
        self.target_abs_mean = stack([x.y for x in self.data_set]).abs().mean().item()
        print("\ntarget_abs_mean %s \n" % self.target_abs_mean, flush=True)
        self.test_target_abs_mean(self.target_abs_mean)
        if self.do_test_output_shape:
            result, message = self.test_output_shape()
            assert result, message
            print(
                "\nVerified that shape of model predictions is equal to shape of labels\n",
                flush=True,
            )

        if self.do_test_input_independent_baseline:
            result, message = self.test_input_independent_baseline()
            assert result, message
            print("Input-independent baseline is verified\n", flush=True)

        if self.do_test_overfit_small_batch:
            result, message = self.test_overfit_small_batch()
            assert result, message
            print(
                f"Verified that a small batch can be overfit since the R2 was greater than {k.DL_DBG_SUFFICIENT_R2_SMALL_BATCH}\n",
                flush=True,
            )

        if self.do_visualize_large_batch_training:
            self.visualize_large_batch_training()

        if self.do_chart_dependencies:
            result, message = self.chart_dependencies()
            assert result, message
            print(
                "Finished charting dependencies. Data is not getting mixed between instances in the same batch.\n",
                flush=True,
            )

        if self.do_choose_model_size_by_overfit:
            best_capacity = self.choose_model_size_by_overfit()
        else:
            best_capacity = None
        print("\nDebug session complete. No errors detected.", flush=True)
        return 