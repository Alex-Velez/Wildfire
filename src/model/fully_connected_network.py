import torch
import torchvision
from torchvision.models import ResNet18_Weights

from model import custom, neural_network

class FullyConnectedNetwork(custom.WildfireModel):
    def __init__(self, source_name: str = ""):
        super().__init__()
        self.model = neural_network._NeuralNetwork()
        self.model_name = "fullyconnectednetwork"
        self.class_names = ["fire", "nofire"]
        self.source_name = source_name

if __name__ == "__main__":
    model = FullyConnectedNetwork()
    print(model.model)