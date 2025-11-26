import time

import torch
from torch import nn
from torch.optim import Adam

from trainer import train_model
from dataloader import WildfireDataLoaders, wildfire_transforms
from wildfiredb import WildFireData1, WildFireData2, WildFireData3, WildFireData4
from models import Resnet18Scratch, ResNet18PreTrained, NeuralNetworkScratch


device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"


def training_runs(models, dataloaders, loss_function, epochs, optimizers, device):
    """Trains each model on each dataset and returns the results of each

    Args:
        models (_type_): _description_
        dataloaders (_type_): _description_
        loss_function (_type_): _description_
        epochs (_type_): _description_
        optimizers (_type_): _description_
        device (_type_): _description_
    """

    for model in models:
        for (train_dataloader, valid_dataloader) in dataloaders:
            print(
                f"Training model {model} on dataset from {train_dataloader.dataset.source}")
            train_model(
                model,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                loss_function=loss_function,
                epochs=epochs,
                optimizer=optimizers[model],
                device=device
            )


wildfire_dl = WildfireDataLoaders(
    [WildFireData1()],
    wildfire_transforms
)
j22_train_dl, j22_valid_dl = wildfire_dl.train_dl, wildfire_dl.valid_dl
print("Training on a dataset of length:", len(j22_train_dl.dataset))
print("Validating on a dataset of length:", len(j22_valid_dl.dataset))

# training hyperparameters
epochs = 10
learning_rate = 0.0001
wf_model = ResNet18PreTrained(j22_train_dl.dataset.source).to(device)
print(f"Starting training on {wf_model}...")
tic = time.time()
train_model(
    wf_model,
    train_dataloader=j22_train_dl,
    valid_dataloader=j22_valid_dl,
    loss_function=nn.CrossEntropyLoss(),
    optimizer=Adam(wf_model.parameters(), lr=learning_rate),
    device=device,
    epochs=epochs,
)
toc = time.time()
print(f"Training complete in {toc - tic} seconds.")
