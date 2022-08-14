import torch

def total_accuracy(output, y):
    # Get the index of the predicted and real
    _, y_pred = torch.max(output.data, dim=1)
    _, y_real = torch.max(y, dim=1)

    return (y_pred == y_real).sum().item() / len(y_real)