import torch
import torch.nn as nn
from face_sense.learn.components import FaceClassifierBasic

def get_train_components(config):
    """Gets the training parameters (model, optimizer, criterion)

    This method creates objects for training/evaluation loop: the model,
    the optimizer and the loss function based on the information on how
    to generate them in the `config` dictionary.

    Note:
        The config dictionary must include `device` key specifying which
        device will the model and the loss function use.

    Args:
        config (dict): The configuration dictionary

    Returns:
        tuple: A tuple containing the components used in training
    """
    # Get the model, optimizer and criterion from the config
    model = build_model(config["model"]).to(config["device"])
    optimizer = build_optimizer(model, config["optimizer"])
    loss_fn = build_criterion(config["criterion"]).to(config["device"])

    return model, optimizer, loss_fn

def build_model(model_info):
    """Builds a model based on the provided parameters.

    This method takes a dictionary with parameters specifying the model
    configuration. The dictionary must at least have a model name, based
    on which the rest of the dictionary key-value pairs are passed to a
    correct model class initializer.

    Args:
        model_info (dict): The model parameters

    Returns:
        nn.Module: A model built on the provided specifications
    
    Raises:
        ValueError: If the name of the model is not recognized
    """
    # Get the model name or use BasicFaceClassifier by default
    model_name = model_info.pop("name", "FaceClassifierBasic")
    
    if model_name == "FaceClassifierBasic":
        return FaceClassifierBasic(**model_info)
    else:
        raise ValueError(f"Model with name {model_name} is invalid!")

def build_optimizer(model, optimizer_info):
    """Builds an optimizer based on the provided parameters.

    This method takes an already created model and a dictionary
    containing optimizer specifications and returns the desired
    optimizer.

    Args:
        model (nn.Module): The model whose parameters to update
        model_info (dict): The optimizer parameters

    Returns:
        torch.optim: An optimizer built on the provided specifications
    
    Raises:
        ValueError: If the name of the optimizer is not recognized
    """
    # Get the optimizer name or use Adam by default
    optimizer_name = optimizer_info.pop("name", "Adam")

    if optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), **optimizer_info)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), **optimizer_info)
    elif optimizer_name == "SGD":
        return torch.optim.SGD(model.parameters(), **optimizer_info)
    else:
        raise ValueError(f"Optimizer with name {optimizer_name} is invalid!")

def build_criterion(criterion_info):
    """Builds a criterion based on the provided parameters.

    This method takes the dictionary with the criterion parameters and
    returns the desired loss function.

    Args:
        criterion_info (dict): The loss parameters

    Returns:
        nn.Module: A criterion built on the provided specifications
    
    Raises:
        ValueError: If the name of the loss function is not recognized
    """
    # Get the criterion name or use CrossEntropyLoss by default
    criterion_name = criterion_info.pop("name", "CrossEntropyLoss")

    if criterion_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(**criterion_info)
    else:
        raise ValueError(f"Criterion with name {criterion_name} is invalid!")
