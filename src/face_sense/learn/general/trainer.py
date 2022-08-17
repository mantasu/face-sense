import os
import copy
import torch

from tqdm import tqdm
from datetime import date
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from face_sense.learn.general.engine import Engine
from face_sense.utils import save_dict, verify_path
from face_sense.learn.tools.config import get_train_components

class Trainer:
    """Trainer that performs the full training/evaluation loops."""

    def __init__(self, config, train_dataset, test_dataset=None):
        """Initializes the Trainer

        Takes the configuration dictionary, the training and optionally
        the test dataset (if not provided, the training will be done on
        k-folds) and trains the model as specified in the configuration
        dictionary.

        Args:
            config (dict): The configuration dictionary with model, data
                loader parameters, training specifications like epochs
                etc. More info in the README.md under *Config* section
                at "train" keyword.
            train_dataset (torch.utils.data.Dataset): The training
                dataset from where inputs and labels will be sampled.
            test_dataset (torch.utils.data.Dataset, optional): The
                testing/evaluation dataset from where inputs and labels
                will be sampled to evaluate the model after every epoch.
                If not provided, it will be a k-fold validation.
                Defaults to None.
        """
        # Assign attributes
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    @staticmethod
    def train(train_loader, test_loader, model, optimizer, loss_fn, **kwargs):
        """A static method to train the model based on the parameters.

        This method takes the parameters to perform training,
        parses additional keyword train arguments, and trains the model
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
        accuracy_fn = kwargs.get("accuracy_name", "total")
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
            kwargs["p_bar_prefix"] = p_bar_prefix + f"({e+1}/{epochs}) "
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
        """Performs k-fold training

        This method takes the full dataset, the model parameters and
        additional arguments for training, splits the dataset into
        k-folds where the current fold is the training set, the other
        folds are the evaluation set and runs a training loop on each.
        Note that on every fold the same model is used thus the model
        overfits. This is the correct case if the dataset is only the
        training dataset or if overfitting is intended, e.g., for face
        recognition.

        Args:
            dataset (dict): The full dataset (or just the training) that
                will be split to k-folds for training.
            *args: The model, the optimizer, and the loss function
            **kwargs: 
        """
        # Parse all the available kwargs parameters
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

        # Initialize history
        fold_performance = []

        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            # Add extra keyword argument for progress bar
            kwargs["p_bar_prefix"] = f"[{fold+1}/{k_folds}] "

            # Sample randomly from a list, no replacement
            train_sampler = SubsetRandomSampler(train_ids)
            test_sampler = SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in fold
            train_loader = DataLoader(dataset, batch_size, sampler=train_sampler)
            test_loader = DataLoader(dataset, batch_size, sampler=test_sampler)

            # Acquire the performance (train/test) history by training
            history = Trainer.train(train_loader, test_loader, *args, **kwargs)
            fold_performance.append({"train": history[0], "test": history[1]})
        
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
        """Prepares the trainer for training

        This method simply prepares the arguments and keyword arguments
        in the `self.config` attribute for training - it creates a
        model, an optimizer etc.

        Returns:
            tuple: A tuple containing a tuple of training arguments
                (train_loader, test_loader, model, optimizer, loss_fn)
                for single training or (dataset, model, optimizer,
                loss_fn) for k-fold training and a dictionary of
                training keyword arguments
        """
        # Initialize config by copying specs config
        config = copy.deepcopy(self.config["specs"])
        
        # Generate the directory/path parameters by reading from "data"
        config.update({
            "is_relative": self.config["data"].get("is_relative", True),
            "model_dir": self.config["data"].get("model_dir", "data/models"),
            "performance_dir": self.config["data"].get("performance_dir", "perf")
        })

        # Generate the training attributes: model, optimizer and loss_fn
        model, optimizer, loss_fn = get_train_components(self.config["params"],
                                                         config["device"])

        if self.test_dataset is not None:
            # Get the data loader params: bs and shuffle
            batch_size = config["specs"].pop("batch_size", 32)
            shuffle = config["specs"].pop("shuffle", True)

            # Create train and test/validation dataset loaders to loop
            train_loader = DataLoader(self.train_dataset, batch_size, shuffle)
            test_loader = DataLoader(self.test_dataset, batch_size, shuffle)

            # Put the main training params to a tuple of arguments
            args = (train_loader, test_loader, model, optimizer, loss_fn)
        else:
            # Put the main training params to a tuple of arguments
            args = (self.train_dataset, model, optimizer, loss_fn)
        
        # Also create a default model name with today's date if needed
        config["model_name"] = self.config["data"].get("model_name", str(date.today()))

        return args, config
    
    def run(self):
        """Simply runs the training.

        It first prepares the training parameters and if there are both
        dataset loaders, it runs the conventional training loop,
        otherwise it runs a k-fold training loop.
        """
        # Prepare the training parameters
        args, kwargs = self.prepare_for_training()

        if len(args) == 5:
            # Simple train if test given
            self.train(*args, **kwargs)
        else:
            # If only train dataset given, k-fold
            self.train_k_folds(*args, **kwargs)
        


        

