import torch

def total_accuracy(output, y):
    """Computes the total accuracy of the predictions given real values.
    
    Takes the model outputs and the real y values (both of shape
    (N, C)), where N - batch size, C - the number of classes and checks
    the index for each sample with the highest confidence and compares
    it with the real index. The computed accuracy is averaged over
    samples.

    Args:
        output (torch.Tensor): The model output of shape (N, C)
        y (torch.Tensor): The real values (one-hot vectors) of shape
            (N, C)

    Returns:
        float: Average total accuracy
    """
    # Get the index of the predicted and real
    _, y_pred = torch.max(output.data, dim=1)
    _, y_real = torch.max(y, dim=1)

    return (y_pred == y_real).sum().item() / len(y_real)