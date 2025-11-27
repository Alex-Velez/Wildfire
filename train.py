import time

import torch
from torch import nn
from torch.optim import Adam

from trainer import train_model
from dataloader import WildfireDataLoaders, wildfire_transforms
from wildfiredb import WildFireData1, WildFireData2, WildFireData3, WildFireData4
from models import WildfireModel, Resnet18Scratch, ResNet18PreTrained, FullyConnectedNetwork


def training_runs(
    models_to_train: list[WildfireModel],
    dataloaders_to_train: list[WildfireDataLoaders],
    loss_function,
    optimizer,
    learning_rate: float = 0.0001,
    epochs: int = 5,
    device: str = "cpu",
):
    """Trains each model on each dataset and returns the results of each

    Args:
        models (WildfireModel): Wildfire model class
        dataloaders (_type_): _description_
        loss_function (_type_): _description_
        epochs (_type_): _description_
        optimizers (_type_): _description_
        device (_type_): _description_
    """

    for model in models_to_train:
        for dataloader in dataloaders_to_train:
            train_dataloader = dataloader.train_dl
            valid_dataloader = dataloader.valid_dl
            model.source_name = train_dataloader.dataset.source
            print(
                f"Training model {model} on dataset from {train_dataloader.dataset.source}")
            model = model.to(device)
            train_model(
                model,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                loss_function=loss_function,
                epochs=epochs,
                optimizer=optimizer(model.parameters(), lr=learning_rate),
                device=device
            )


if __name__ == "__main__":
    device = torch.accelerator.current_accelerator(
    ).type if torch.accelerator.is_available() else "cpu"

    # training hyperparameters
    epochs = 5
    learning_rate = 0.0001
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam  # optimized must be initialized per model

    # prepare dataloaders
    wildfire_dl = [
        # training on one dataset
        WildfireDataLoaders([WildFireData4()], wildfire_transforms),
        # training on a combination of datasets
        WildfireDataLoaders([WildFireData1(), WildFireData2()], wildfire_transforms),
        # # training on all datasets
        WildfireDataLoaders([WildFireData1(), WildFireData2(),
                             WildFireData3(), WildFireData4()], wildfire_transforms),]

    # models to train
    models = [
        FullyConnectedNetwork(),
        Resnet18Scratch(),
        ResNet18PreTrained(),]

    training_runs(
        models,
        wildfire_dl,
        loss_function,
        epochs=epochs,
        optimizer=optimizer,
        device=device
    )
