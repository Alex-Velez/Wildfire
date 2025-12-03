import time

import torch
from torch import nn
from torch.optim import Adam, SGD

from trainer import train_model
from dataloader import WildfireDataLoaders, wildfire_transforms
from wildfiredb import WildFireData1, WildFireData2, WildFireData3, WildFireData4
from models import WildfireModel, Resnet34Scratch, ResNet18PreTrained, FullyConnectedNetwork


def training_runs(
    models_to_train: list[WildfireModel],
    dataloaders_to_train: list[WildfireDataLoaders],
    loss_function,
    optimizer,
    learning_rate: float = 0.0001,
    epochs: int = 5,
    device: str = "cpu",
) -> dict:
    """Trains each model on each dataset and returns the results of each

    Args:
        models (WildfireModel): Wildfire model class
        dataloaders (_type_): _description_
        loss_function (_type_): _description_
        epochs (_type_): _description_
        optimizers (_type_): _description_
        device (_type_): _description_
    """

    model_results = {}

    for model in models_to_train:
        for dataloader in dataloaders_to_train:
            train_dataloader = dataloader.train_dl
            valid_dataloader = dataloader.valid_dl

            model_to_train = model(
                source_name=train_dataloader.dataset.source).to(device)
            print(
                f"Training model {model_to_train} on dataset from {train_dataloader.dataset.source}")
            train_loss, train_acc, val_loss, val_acc = train_model(
                model_to_train,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                loss_function=loss_function,
                epochs=epochs,
                optimizer=optimizer(
                    model_to_train.parameters(), lr=learning_rate),
                device=device,
                early_stopping=False
            )

            model_results[str(model_to_train)] = {} if model_results.get(
                str(model_to_train)) is None else model_results[str(model_to_train)]
            model_results[str(model_to_train)][train_dataloader.dataset.source] = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "valid_loss": val_loss,
                "valid_accuracy": val_acc
            }

    return model_results


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = torch.accelerator.current_accelerator(
    ).type if torch.accelerator.is_available() else "cpu"

    # training hyperparameters
    epochs = 50
    learning_rate = 0.0001
    loss_function = nn.CrossEntropyLoss()
    optimizer = SGD  # optimizer must be initialized per model

    # prepare dataloaders
    wildfire_dl = [
        # training on one dataset
        WildfireDataLoaders([WildFireData4()], wildfire_transforms),
        # # training on all datasets
        WildfireDataLoaders(
            [WildFireData1(), WildFireData4()], wildfire_transforms),
    ]

    # models to train, must be initialized per training run
    models = [
        FullyConnectedNetwork,
        Resnet34Scratch,
        # ResNet18PreTrained,
    ]

    training_results = training_runs(
        models,
        wildfire_dl,
        loss_function,
        epochs=epochs,
        optimizer=optimizer,
        device=device
    )

    # Example: Plot training and validation accuracy for each model and dataset and save
    for model_name, datasets in training_results.items():
        for dataset_name, metrics in datasets.items():
            plt.figure(figsize=(10, 5))
            plt.plot(metrics['train_accuracy'], label='Training Accuracy')
            plt.plot(metrics['valid_accuracy'], label='Validation Accuracy')
            plt.title(f'Accuracy for {model_name} on {dataset_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.plot(metrics['train_loss'], label='Training Loss')
            plt.plot(metrics['valid_loss'], label='Validation Loss')
            plt.title(f'Loss for {model_name} on {dataset_name}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    print("Training complete. Results:")
    print(training_results)

    import json
    file = "training_results.json"
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(training_results, f, indent=4)
