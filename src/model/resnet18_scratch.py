import torch
import torchvision

from model import custom

class Resnet18Scratch(custom.WildfireModel):
    def __init__(self, source_name: str = ""):
        super().__init__()
        
        self.model = torchvision.models.resnet18(weights=None)
        
        # modify the final layer to have two classes
        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            len(self.class_names)
        )
        
        self.model_name = "resnet18scratch"
        self.class_names = ["fire", "nofire"]
        self.source_name = source_name

if __name__ == "__main__":
    model = Resnet18Scratch()
    print(model.model)