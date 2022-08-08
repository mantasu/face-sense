import os
import torch
import copy

from tqdm import tqdm
from datetime import date
from sklearn.model_selection import KFold
from face_sense.utils import save_dict, verify_path
from face_sense.learn.tools.engine import Engine
from face_sense.learn.tools.helper import get_train_components
from torch.utils.data import DataLoader, SubsetRandomSampler

class Trainer:
    def __init__(self, config, train_dataset, test_dataset=None):
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    @staticmethod
    def train(train_loader, test_loader, model, optimizer, loss_fn, **kwargs):
        """A static method to train the model based on the parameters.

        This method takes the parameters to perform training,
        initializes default accuracy functions, and trains the model
        for the provided number of epochs, at each epoch calling the
        :func:`~engine.Engine.train` method and its evaluation method
        on the eval/test dataset. This method can optionally save the
        model state and save the performance histories to a separate
        JSON file at the end of training.

        Args:
            train_loader (torch.utils.data.DataLoader): The train
                loader with input and label values
            test_loader (torch.utils.data.DataLoader): The val/test
                loader with input and label values
            model (nn.Module): The model for the data
            optimizer (torch.optim.Optimizer): The optimizer
            loss_fn (nn.Module): The loss function
            **kwargs: Additional parameters for training

        Returns:
            tuple: Dicts of train and val/test performance histories
        """
        # Extract keyword method arguments
        epochs = kwargs.pop("epochs", 1)
        device = kwargs.pop("device", "cpu")
        model_name = kwargs.pop("model_name", "na")
        p_bar_prefix = kwargs.pop("p_bar_prefix", "")
        accuracy_fn = kwargs.get("accuracy_fn", "total")
        model_dir = kwargs.pop("model_dir", None)
        is_relative = kwargs.pop("is_relative", True)
        performance_dir = kwargs.pop("performance_dir", None)

        # Prepare the progress bar over es
        progress_bar = tqdm(range(epochs))

        # Initialize the engine for training and evaluation
        engine = Engine(model, optimizer, loss_fn, device)

        if not isinstance(accuracy_fn, list):
            # To list for convenience
            accuracy_fn = [accuracy_fn]
        
        # Init history
        performance = {
            "train": {key: [] for key in accuracy_fn + ["loss"]},
            "test": {key: [] for key in accuracy_fn + ["loss"]}
        }

        for e in progress_bar:
            # Update kwargs related to current progress bar
            kwargs["p_bar_prefix"] = p_bar_prefix + f"({e}/{epochs}) "
            kwargs["progress_bar"] = progress_bar

            # Training and evaluation iterations and get performance
            train_performance = engine.train(train_loader, **kwargs)
            test_performance = engine.eval(test_loader, **kwargs)

            # Description
            desc = {}

            for key in train_performance.keys():
                # Append performance values to the performance history
                performance["train"][key].append(train_performance[key])
                performance["test"][key].append(test_performance[key])
                desc[key] = f"{train_performance[key]}|{test_performance[key]}"
            
            if not isinstance(progress_bar, int):
                # Update progress bar description and postfix information
                progress_bar.set_description(p_bar_prefix + f"({e}/{epochs})")
                progress_bar.set_postfix(desc)
        
        if performance_dir is not None:
            # Generate path and save the performance in a .json file
            path = os.path.join(performance_dir, model_name + ".json")
            path = verify_path(path, is_relative)
            save_dict(performance, path)
        
        if model_dir is not None:
            # Generate path and save the model in a .pth file
            path = os.path.join(model_dir, model_name + ".pth")
            path = verify_path(path, is_relative)
            torch.save(model.state_dict(), path)
        
        return performance["train"], performance["test"]
    
    @staticmethod
    def train_k_folds(dataset, *args, **kwargs):
        batch_size = kwargs.pop("batch_size", 32)
        k_folds = kwargs.pop("k_folds", 5)
        seed = kwargs.pop("seed", 42)
        performance_dir = kwargs.pop("performance_dir", "data/performance")
        model_dir = kwargs.pop("model_dir", "data/models")
        is_relative = kwargs.get("is_relative", True)
        model_name = kwargs.get("model_name", "na")
        shuffle = kwargs.pop("shuffle", True)
        
        if shuffle:
            # Initiate the k-fold object which will sample randomly
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        else:
            # Initiate the k-fold object
            kfold = KFold(n_splits=k_folds)

        fold_performance = []

        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            # Add extra keyword argument for progress bar
            kwargs["p_bar_prefix"] = f"[{fold}/{k_folds}] "

            # Sample randomly from a list, no replacement
            train_sampler = SubsetRandomSampler(train_ids)
            test_sampler = SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in fold
            train_loader = DataLoader(dataset, batch_size, sampler=train_sampler)
            test_loader = DataLoader(dataset, batch_size, sampler=test_sampler)

            # Acquire the performance (train/test) history by training
            history = Trainer.train(train_loader, test_loader, *args, **kwargs)
            fold_performance = {"train": history[0], "test": history[1]}
        
        if performance_dir is not None:
            # Generate path and save the performance in a .json file
            path = os.path.join(performance_dir, model_name + ".json")
            path = verify_path(path, is_relative)
            save_dict(fold_performance, path)
        
        if model_dir is not None:
            # Generate path and save the model in a .pth file
            path = os.path.join(model_dir, model_name + ".pth")
            path = verify_path(path, is_relative)
            torch.save(args[0].state_dict(), path)
    
    def prepare_for_training(self):
        # Copy dict and generate model prams
        config = copy.deepcopy(self.config)
        model, optimizer, loss_fn = get_train_components(config)

        # Delete unnecessary
        del config["model"]
        del config["optimizer"]
        del config["criterion"]

        if self.test_dataset is not None:
            # Get the data loader params: bs and shuffle
            batch_size = config.pop("batch_size", 32)
            shuffle = config.pop("shuffle", True)

            # Create train and test/validation dataset loaders to loop
            train_loader = DataLoader(self.train_dataset, batch_size, shuffle)
            test_loader = DataLoader(self.test_dataset, batch_size, shuffle)

            # Put the main training params to a tuple of arguments
            args = (train_loader, test_loader, model, optimizer, loss_fn)
        else:
            # Put the main training params to a tuple of arguments
            args = (self.train_dataset, model, optimizer, loss_fn)
        
        # Also create a default model name with today's date if needed
        config["model_name"] = config.get("model_name", date.today())

        return args, config
    
    def run(self):
        # Prepare the training parameters
        args, kwargs = self.prepare_for_training()

        if len(args) == 5:
            # Sole train if test given
            self.train(*args, **kwargs)
        else:
            # If only train dataset given, k-fold
            self.train_k_folds(*args, **kwargs)
        


        

