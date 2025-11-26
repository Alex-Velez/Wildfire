"""Each Model (WildfireModel) will have a .model attribute that can be used for inference.
"fire" is labelled as 1 and "nofire" as 0.
Expecting to have one model for each dataset and one model combining all datasets.
"""

import torch

from wildfiredb import WildFireData1
from dataloader import WildfireDataLoaders, wildfire_transforms
from models import WildfireModel


device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"


if __name__ == "__main__":
    model = WildfireModel().to(device)
    test_dl = WildfireDataLoaders(
        [WildFireData1()],
        wildfire_transforms
    ).test_dl

    # make one inference and decode the label
    model.eval()
    with torch.no_grad():
        for images, labels in test_dl:
            images = images.to(device)
            outputs = model.model(images)
            preds = outputs.argmax(dim=1)
            print("Predicted labels:", model.decode_labels(preds.cpu().numpy()))
            print("True labels:", model.decode_labels(labels.cpu().numpy()))
            break  # only one batch
