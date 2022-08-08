import torch.nn as nn

class FaceClassifierBasic(nn.Module):
    def __init__(self, in_shape, num_classes):
        super().__init__()

        self.layer1 = self._create_block(in_shape, 1024)
        self.layer2 = self._create_block(1024, 1024)
        self.layer3 = nn.Linear(1024, num_classes)
    
    def _create_block(self, in_shape, out_shape):
        linear = nn.Sequential(
            nn.Linear(in_shape, out_shape),
            nn.ReLU(),
            nn.Dropout(0.5))
        
        return linear

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x