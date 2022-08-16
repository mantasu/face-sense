import torch
from tqdm import tqdm
from face_sense.learn.tools.accuracy import *

class Engine():
    """Engine class that performs training and evaluation."""

    ACCURACY_MAP = {
        "total": total_accuracy
    }

    def __init__(self, model, optimizer, loss_fn, device):
        """Initializes the engine

        Args:
            model (torch.nn.Module): The model for the data
            optimizer (torch.optim.Optimizer): The optimizer
            loss_fn (torch.nn.Module): The loss function
            accuracy_fn (str): The name of the accuracy function
            device (torch.device): The device to use for training
        """
        # Initialize engine parameters
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
    
    def train(self, data_loader, **kwargs):
        """Performs a training step for one epoch.

        This method takes the data loader to run the model training on,
        and the accuracy functions which should be run on full training
        set after one epoch training.

        Args:
            data_loader (torch.utils.data.Dataloader): The data loader
                to run the training on.
            accuracy_fn (str|list): 
            **kwargs: The additional arguments for training:
                * progress_bar (tqdm, optional): The progress bar to
                    update during training/evaluation. If not provided,
                    no progress bar will be updated. Defaults to None.
                * update_every (int, optional): Every n iterations to
                    update the running loss/accuracy. Defaults to 10.
                * p_bar_prefix (str, optional): The prefix text for the
                    progress bar. Defaults to "".
                * accuracy_fn (str|list, optional): The accuracy(-ies)
                    to check after the parameter updates after one epoch
                    if evaluation is desired. Defaults to "total".
                * require_accuracy (bool, optional): Whether to compute
                    running accuracy. Defaults to True.
                * require_evaluation (bool, optional): Whether to
                    evaluate the model on the training dataset after one
                    epoch. Defaults to True.

        Returns:
            dict: A dictionary with accuracies
        """
        progress_bar = kwargs.get("progress_bar", 0)
        p_bar_prefix = kwargs.get("p_bar_prefix", "")
        update_every = kwargs.get("update_every", 10)
        accuracy_fn = kwargs.get("accuracy_name", "total")
        require_accuracy = kwargs.get("require_accuracy", True)
        require_evaluation = kwargs.get("require_evaluation", True)

        # Init train mode and running loss
        self.model.train()
        running_loss = 0
        running_acc = 0

        if progress_bar is not None:
            # Initialize the progress bar description with loss and acc
            progress_bar.set_description(f"{p_bar_prefix}Loss: - Acc: - %")

        for i, (x, y) in enumerate(data_loader):
            # Cast to correct device
            x = x.to(self.device)
            y = y.to(self.device)

            # Restore gradient and compute output
            self.optimizer.zero_grad()
            output = self.model(x)

            # Compute model loss and backpropagate
            loss = self.loss_fn(output, y)
            loss.backward()

            # Parameter and loss update
            self.optimizer.step()
            running_loss += loss.item()

            if require_accuracy:
                # Compute the running batch accuracy
                running_acc += total_accuracy(output, y)

            if self.device == "cuda:0":
                # Free up GPU memory space
                torch.cuda.empty_cache()
            
            if (i + 1) % update_every == 0:
                # Compute the average batch loss and accuracy
                batch_loss = f"{running_loss / update_every:.6f}"
                batch_acc = f"{running_acc / update_every * 100:.2f} %"
                batch_acc = batch_acc if require_accuracy else '-'
                
                if progress_bar is not None:
                    # Update the progress bar with avg batch performance
                    description = f"Loss: {batch_loss} Acc: {batch_acc}"
                    progress_bar.set_description(p_bar_prefix + description)
                
                # Empty running
                running_loss = 0
                running_acc = 0
        
        if require_evaluation:
            # Get the final performance over the training set
            performance = self.eval(data_loader, accuracy_name=accuracy_fn)
        else:
            if not isinstance(accuracy_fn, list):
                # Make it a list for readability
                accuracy_fn = [accuracy_fn]
            
            # Generate a dummy performance dictionary
            performance = {acc: None for acc in accuracy_fn}

        return performance

    def eval(self, data_loader, **kwargs):
        """Evaluates the model on the provided data loader.

        This method takes the data loader, performs forward pass on it
        and finds the accuracies for each desired type.

        Args:
            data_loader (torch.utils.data.Dataloader): The data loader
                to run the evaluation on.
            accuracy_fn (str|list): The accuracy(-ies) to check after
                the parameter updates after one epoch.

        Returns:
            dict: A dictionary of the performances for each accuracy
        """
        accuracy_fn = kwargs.get("accuracy_name", "total")

        if isinstance(accuracy_fn, str):
            # Generalize to single list
            accuracy_fn = [accuracy_fn]
    
        # Switch to eval
        self.model.eval()

        # Initialize the prediction and label lists (tensors)
        ys_pred, ys_real = [], []

        # Initialize loss
        final_loss = 0

        with torch.no_grad():
            for x, y in data_loader:
                # Cast to device
                x = x.to(self.device)
                y = y.to(self.device)

                # Compute output
                output = self.model(x)

                # Compute the current loss
                loss = self.loss_fn(output, y)
                final_loss += loss.item()

                # Append pred and real
                ys_pred.append(output)
                ys_real.append(y)

                if self.device == "cuda:0":
                    # Free up GPU memory space
                    torch.cuda.empty_cache()

            # Concat pred and real values
            y_pred = torch.concat(ys_pred)
            y_real = torch.concat(ys_real)
        
        # Calculate an accuracy score for each provided accuracy type
        scores = {key: round(self.ACCURACY_MAP[key](y_pred, y_real), 2) for key in accuracy_fn}
        scores["loss"] = round(final_loss / len(data_loader), 4)
        
        return scores

