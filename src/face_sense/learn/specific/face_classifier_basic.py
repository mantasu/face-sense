import torch.nn as nn

class FaceClassifierBasic(nn.Module):
    """Simple Face Classifier with a custom number of hidden layers"""

    def __init__(self, in_shape, num_classes, hidden_shape=[1024, 1024]):
        """Initializes teh face classifier.

        Takes in the attributes and initializes the neural blocks for
        training or for loading the parameters for inference.

        Args:
            in_shape (int): The embedding size
            num_classes (int): The number of recognizable identities
            hidden_shape (list, optional): The list of hidden units
                defining the structure over the hidden layers.
                Defaults to [1024, 1024].
        """
        super().__init__()

        # Create a list of input shapes for layers
        in_shapes = [in_shape] + hidden_shape[:-1]
        blocks = []

        for num_in, num_out in zip(in_shapes, hidden_shape):
            # Create a linear-relu-dropout sequential block and append
            blocks.append(self._create_block(num_in, num_out))
        
        # Init a sequential nn, linear layer
        self.blocks = nn.Sequential(*blocks)
        self.last = nn.Linear(hidden_shape[-1], num_classes)
    
    def _create_block(self, in_shape, out_shape):
        """Creates a feed-forward NN block.

        Takes the number of input neurons and the number of output
        neurons, and creates a NN block with linear-relu-dropout
        sub-layers.

        Args:
            in_shape (int): The number of input neurons
            out_shape (int): The number of output neurons

        Returns:
            torch.nn.Sequential: A sequence of sub-layers
        """
        linear = nn.Sequential(
            nn.Linear(in_shape, out_shape),
            nn.ReLU(),
            nn.Dropout(0.5))
        
        return linear

    def forward(self, x):
        """Performs a forward operation.

        Takes an input of shape (N, D), where N - the batch size, D -
        the embedding size and performs a forward pass based on the
        initialized blocks.

        Args:
            x (torch.Tensor): An input tensor of shape (N, D)

        Returns:
            torch.Tensor: Network outputs (logits) of shape (N, C)
        """        
        # Last forward pass
        x = self.blocks(x)
        x = self.last(x)

        return x