import torch
import torchvision.transforms.v2
from torchvision.transforms.v2 import Compose, ToDtype, Lambda, Resize, Normalize

import data_loader
import database
from model import fully_connected_network, resnet18_pretrained, resnet18_scratch
import train

wildfire_transforms = Compose([
    ToDtype(torch.float32),
    Lambda(lambda img: img / 255.0),  # clamp values to [0, 1]
    Resize([224, 224]),
    # ImageNet stats
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),  
])

de_wildfire_transforms = Compose([
    Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    ),
    Lambda(lambda img: img * 255.0),
    ToDtype(torch.uint8),
])


if __name__ == "__main__":
    device = "cpu" if not torch.accelerator.is_available() else torch.accelerator.current_accelerator()
    
    # training hyperparameters
    epochs = 5
    learning_rate = 0.0001
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam  # optimized must be initialized per model

    # prepare dataloaders
    wildfire_data_loaders = [
        # training on one dataset
        data_loader.WildfireDataLoaders(
            [database.RawDataPaths4()],
            wildfire_transforms
        ),
        
        # training on a combination of datasets
        data_loader.WildfireDataLoaders(
            [database.RawDataPaths1(), database.RawDataPaths2()], 
            wildfire_transforms
        ),
        
        # training on all datasets
        # data_loader.WildfireDataLoaders(
        #     [
        #         database.RawDataPaths1(),
        #         database.RawDataPaths2(),
        #         database.RawDataPaths3(),
        #         database.RawDataPaths4()
        #     ],
        #     wildfire_transforms
        # ),
    ]

    # models to train
    models = [
        fully_connected_network.FullyConnectedNetwork(),
        resnet18_scratch.Resnet18Scratch(),
        resnet18_pretrained.ResNet18PreTrained(),
    ]

    train.training_runs(
        models,
        wildfire_data_loaders,
        loss_function,
        epochs=epochs,
        optimizer=optimizer,
        device=str(device)
    )