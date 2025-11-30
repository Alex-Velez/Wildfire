import torch
import tqdm

import model.custom
import data_loader
import paths


def train_epoch(model, dataloader, loss_function, optimizer, device):
    """One epoch of training for the model on the dataloader data.

    Args:
        model (nn.Module): Model to perform training on
        dataloader (_type_): Training data as torch dataloader
        loss_function (_type_): Loss function
        optimizer (_type_): Optimization Algorithm to use for weight update
        device (_type_): Device to run on

    Returns:
        tuple: Average loss, accuracy for the epoch
    """
    model.train()  # set model to training mode

    total_loss = 0.0
    correct_predictions = 0
    total = 0

    for images, labels in tqdm.tqdm(dataloader):
        images, true_labels = images.to(device), labels.to(device)

        # forward pass
        optimizer.zero_grad()
        outputs = model.model(images)
        loss = loss_function(outputs, true_labels)

        # backward pass
        loss.backward()
        optimizer.step()

        # track progress
        total_loss += loss.item() * images.size(0)
        pred_labels = outputs.argmax(dim=1)
        correct_predictions += (pred_labels == true_labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct_predictions / total
    return avg_loss, accuracy


def validate(model, dataloader, loss_function, device):
    """Goes over the validation set and returns loss and accuracy

    Args:
        model (_type_): Trained Model
        dataloader (_type_): Dataloader to perform 
        loss_function (_type_): _description_
        device (_type_): _description_

    Returns:
        tuple: validation loss, validation accuracy
    """
    # set the model to eval mode
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in tqdm.tqdm(dataloader):
        images, true_labels = images.to(device), labels.to(device)

        # calculate loss
        pred_outputs = model.model(images)
        loss = loss_function(pred_outputs, true_labels)

        # required since last batch may not be same size
        total_loss += loss.item() * images.size(0)
        pred_labels = pred_outputs.argmax(dim=1)
        correct_predictions += (pred_labels == true_labels).sum().item()
        total_predictions += images.size(0)

    validation_loss = total_loss / total_predictions
    validation_accuracy = correct_predictions / total_predictions

    return validation_loss, validation_accuracy


def train_model(model, train_dataloader, valid_dataloader, loss_function, optimizer, device, epochs=1):
    """Controls the training and validation process."""
    best_valid_acc = 0.0
    previous_valid_acc = 0.0

    # for visualization
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(
            model, train_dataloader, loss_function, optimizer, device)
        print(
            f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")

        valid_loss, valid_acc = validate(
            model, valid_dataloader, loss_function, device)
        print(
            f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}")

        # save the best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

            torch.save(model.state_dict(), paths.PARAMS_DIR / f"{model}.pth")
            print("Best model saved.")
            

        # early stopping condition
        if valid_acc > 0.9999:
            print("Early stopping triggered, validation accuracy reached ~100%.")
            break

        if (valid_acc - previous_valid_acc) < 0.001:
            print(
                "Early stopping triggered, validation accuracy stopped increasing (significantly).")
            break

        previous_valid_acc = valid_acc


def training_runs(
    models_to_train: list[model.custom.WildfireModel],
    dataloaders_to_train: list[data_loader.WildfireDataLoaders],
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
            print(f"Training model {model} on dataset from {train_dataloader.dataset.source}")
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

