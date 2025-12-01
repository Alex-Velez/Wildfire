"""Contains the models available for training.

"""
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights


class WildfireModel(nn.Module):
    """Base class for Wildfire Models

    Args:
        nn (_type_): _description_
    """

    def __init__(self, source_name: str = ""):
        super().__init__()
        self.model: nn.Module = None
        self.class_names = ["fire", "nofire"]
        self.source_name = source_name

    def decode_labels(self, indices):
        if isinstance(indices, int):
            return self.class_names[indices]
        return [self.class_names[i] for i in indices]
    
    def __str__(self):
        return self.model_name + "_" + self.source_name


class ResNet18PreTrained(WildfireModel):
    def __init__(self, source_name: str = ""):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # modify the final layer to have two classes
        self.model.fc = nn.Linear(
            self.model.fc.in_features, len(self.class_names))
        self.model_name = "resnet18pretrained"
        self.source_name = source_name


class Resnet34Scratch(WildfireModel):
    def __init__(self, source_name: str = ""):
        super().__init__()
        self.model = resnet34(weights=None)
        # modify the final layer to have two classes
        self.model.fc = nn.Linear(
            self.model.fc.in_features, len(self.class_names))
        self.model_name = "resnet34scratch"
        self.source_name = source_name


class _NeuralNetwork(nn.Module):
    """Fully connected Neural Network from Pytorch tutorial
    https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(224*224*3, 512),  # modified for RGBimages
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class FullyConnectedNetwork(WildfireModel):
    def __init__(self, source_name: str = ""):
        super().__init__()
        self.model = _NeuralNetwork()
        self.model_name = "fullyconnectednetwork"
        self.source_name = source_name


if __name__ == "__main__":
    model = Resnet34Scratch()
    print(model.model)
