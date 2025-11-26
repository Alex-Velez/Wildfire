from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18PreTrained(nn.Module):
    def __init__(self, dataset_source: str = ""):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.class_names = ["fire", "nofire"]
        # modify the final layer to have two classes
        self.model.fc = nn.Linear(
            self.model.fc.in_features, len(self.class_names))
        self.dataset_source = dataset_source

    def decode_labels(self, indices):
        if isinstance(indices, int):
            return self.class_names[indices]
        return [self.class_names[i] for i in indices]

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred.argmax(dim=1)  # prediction with highest score

    def __str__(self):
        return "resnet18finetuned_" + self.dataset_source


class Resnet18Scratch(nn.Module):
    def __init__(self, dataset_source: str = ""):
        super().__init__()
        self.model = resnet18(weights=None)
        self.class_names = ["fire", "nofire"]
        # modify the final layer to have two classes
        self.model.fc = nn.Linear(
            self.model.fc.in_features, len(self.class_names))
        self.dataset_source = dataset_source

    def decode_labels(self, indices):
        if isinstance(indices, int):
            return self.class_names[indices]
        return [self.class_names[i] for i in indices]

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred.argmax(dim=1)  # prediction with highest score

    def __str__(self):
        return "resnet18scratch_" + self.dataset_source

class NeuralNetwork(nn.Module):
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
    

class NeuralNetworkScratch(nn.Module):
    def __init__(self, dataset_source: str = ""):
        super().__init__()
        self.model = NeuralNetwork()
        self.class_names = ["fire", "nofire"]
        # modify the final layer to have two classes
        # self.model.fc = nn.Linear(
        #     self.model.fc.in_features, len(self.class_names))
        self.dataset_source = dataset_source

    def decode_labels(self, indices):
        if isinstance(indices, int):
            return self.class_names[indices]
        return [self.class_names[i] for i in indices]

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred.argmax(dim=1)  # prediction with highest score

    def __str__(self):
        return "neuralnetworkscratch_" + self.dataset_source


if __name__ == "__main__":
    model = NeuralNetworkScratch()
    print(model)
