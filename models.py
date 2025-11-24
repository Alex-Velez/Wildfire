from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class WildfireModel(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(WildfireModel, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred.argmax(dim=1)
