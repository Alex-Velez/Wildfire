import torch
from torch.nn import Flatten, Sequential, Linear, ReLU

class _NeuralNetwork(torch.nn.Module):
    """Fully connected Neural Network from Pytorch tutorial
    https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        
        self.flatten = Flatten()
        self.linear_relu_stack = Sequential(
            Linear(224 * 224 * 3, 512),  # modified for RGBimages
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    model = _NeuralNetwork()
    print(model.model)