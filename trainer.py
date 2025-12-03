"""Trains a model given a training and validation dataloader, los function and optimizer.
    
TODO: Decide how to record and show training results.
"""
import os
import torch
from tqdm import tqdm


def train_epoch(model, dataloader, loss_function, optimizer, device):
    """One epoch of training for the model on the dataloader data.

    Args:
        model (nn.Module): Model to perform training on
        dataloader (_type_): Training data as torch dataloader
        loss_function (_type_): Loss function
        optimizer (_type_): Optimzation Algorithm to use for weight update
        device (_type_): Device to run on

    Returns:
        tuple: Average loss, accuracy for the epoch
    """
    model.train()  # set model to training mode

    total_loss = 0.0
    correct_preds = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images, true_labels = images.to(device), labels.to(device)

        # forward pass
        optimizer.zero_grad()
        outputs = model.model(images)
        loss = loss_function(outputs, true_labels)

        # backward pass
        loss.backward()   # calculate gradients
        optimizer.step()  # update weights

        # track progress
        total_loss += loss.item() * images.size(0)
        pred_labels = outputs.argmax(dim=1)
        correct_preds += (pred_labels == true_labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct_preds / total
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
    correct_preds = 0
    total_preds = 0

    for images, labels in tqdm(dataloader):
        images, true_labels = images.to(device), labels.to(device)

        # calculte loss
        pred_outputs = model.model(images)
        loss = loss_function(pred_outputs, true_labels)

        total_loss += loss.item() * images.size(0)
        pred_labels = pred_outputs.argmax(dim=1)
        correct_preds += (pred_labels == true_labels).sum().item()
        # required since last batch may not be same size
        total_preds += images.size(0)

    validation_loss = total_loss / total_preds
    validation_accuracy = correct_preds / total_preds

    return validation_loss, validation_accuracy


def train_model(model, train_dataloader, valid_dataloader, loss_function, optimizer, device, epochs=1, early_stopping=True) -> list[str]:
    """Controls the training and validation process and saves the best model parameters.
    
    Returns the training and validation losses and accuracies for each epoch."""
    best_valid_acc = 0.0
    previous_valid_acc = 0.0

    # for visualization
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(
            model, train_dataloader, loss_function, optimizer, device)
        print(
            f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        valid_loss, valid_acc = validate(
            model, valid_dataloader, loss_function, device)
        print(
            f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}")
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        # save the best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

            params_dir = "model_params/"
            params_path = os.path.dirname(__file__) + "/" + params_dir
            torch.save(model.state_dict(), f"{params_path}{model}.pth")
            print("Best model saved.")

        # early stopping condition
        if early_stopping:
            if valid_acc > 0.9999:
                print("Early stopping triggered, validation accuracy reached ~100%.")
                break

            if (valid_acc - previous_valid_acc) < 0.001:
                print(
                    "Early stopping triggered, validation accuracy stopped increasing (significantly).")
                break

        previous_valid_acc = valid_acc

    return train_losses, train_accuracies, valid_losses, valid_accuracies
    