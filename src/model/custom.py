import torch
from torchvision.models import resnet18, ResNet18_Weights


class WildfireModel(torch.nn.Module):
    """Base class for Wildfire Models

    Args:
        nn (_type_): _description_
    """

    def __init__(self, source_name: str = ""):
        super().__init__()
        
        self.model: torch.nn.Module = torch.nn.Module() # None
        self.class_names = ["fire", "nofire"]
        self.source_name = source_name

    def decode_labels(self, indices):
        if isinstance(indices, int):
            return self.class_names[indices]
        return [self.class_names[i] for i in indices]

    def forward(self, x):
        y_pred = self.model(x)  # type: ignore
        return y_pred.argmax(dim=1)  # prediction with highest score

    def __str__(self) -> str:
        # return self.model_name + "_" + self.source_name
        return self.model._get_name() + "_" + self.source_name

if __name__ == "__main__":
    model = WildfireModel()
    print(model.model)