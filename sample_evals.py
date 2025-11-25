"""Each Model (WildfireModel) will have a .model attribute that can be used for inference.
"fire" is labelled as 1 and "nofire" as 0.
Expecting to have one model for each dataset and one model combining all datasets.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from dataloader import test_dl
from models import WildfireModel


device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"


def evaluate(model: torch.nn.Module, dataloader, device: str):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)                  # images: (B, C, H, W)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
            outputs = model.model(images)               # note: WildfireModel wraps resnet as .model
            batch_preds = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds.tolist())
            targets.extend(labels.cpu().numpy().tolist())

    preds = np.array(preds)
    targets = np.array(targets)
    acc = accuracy_score(targets, preds)
    cm = confusion_matrix(targets, preds)
    report = classification_report(targets, preds, target_names=["fire", "nofire"], digits=4)
    return acc, cm, report

if __name__ == "__main__":
    model = WildfireModel(num_classes=2).to(device)

    # acc, cm, report = evaluate(model, test_dl, device)
    # print(f"Accuracy: {acc:.4f}")
    # print("Confusion matrix:\n", cm)
    # print("Classification report:\n", report)
    
    # make one inference and decode the label
    model.eval()
    with torch.no_grad():
        for images, labels in test_dl:
            images = images.to(device)
            outputs = model.model(images)
            preds = outputs.argmax(dim=1)
            decoded_labels = model.decode_labels(preds.cpu().numpy())
            print("Decoded labels:", decoded_labels)
            break  # only one batch
