import torch
import torchvision
from torchvision.models import ResNet18_Weights

from model import custom

class ResNet18PreTrained(custom.WildfireModel):
    def __init__(self, source_name: str = ""):
        super().__init__()
        
        self.model = torchvision.models.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )
        
        # modify the final layer to have two classes
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            len(self.class_names)
        )
        
        self.model_name = "resnet18pretrained"

        self.class_names = ["fire", "nofire"]
        self.source_name = source_name


if __name__ == "__main__":
    model = ResNet18PreTrained()
    print(model.model)