import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision.transforms import v2

from wildfiredb import dataframe_train_partials, dataframe_test_partials, dataframe_valid_partials


# Placeholder


transforms = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.Lambda(lambda img: img / 255.0),  # clamp values to [0, 1]
    v2.Resize([224, 224]),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),  # ImageNet stats
])
de_transforms = v2.Compose([
    v2.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    ),
    v2.Lambda(lambda img: img * 255.0),
    v2.ToDtype(torch.uint8),
])


class DatasetWildfire(Dataset):
    """Transforms Wildfire dataframe into a PyTorch Dataset.
    Label "fire" is mapped to 1, "nofire" to 0.
        

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataframe: pd.DataFrame, transforms):
        """
        Args:
            dataframe (pandas.DataFrame): 
                Columns - "Dataset" | "Image" | "Class"
                "ImagePath" - relative path to the image file
                "Image" - Path to image file
                "Inferred" - inferred label from previous model (if any)
                "Class" - "fire" or "nofire"
            transforms (torchvision.transforms): 
                image preprocessing steps defined in torchvision.transforms
        """
        self.dataframe = dataframe
        self.transforms = transforms
        self.class_to_idx = {"fire": 1, "nofire": 0}

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_file = self.dataframe.iloc[idx]["ImagePath"]
        img_file = os.path.dirname(os.path.abspath(__file__)) / img_file
        img_tensor = decode_image(img_file)
        label_str = self.dataframe.iloc[idx]['Class']
        label = self.class_to_idx[label_str]  # convert to integer

        img_transformed = self.transforms(img_tensor)

        return img_transformed, label


dataframe_train = pd.concat(dataframe_train_partials, ignore_index=True)
dataframe_test = pd.concat(dataframe_test_partials, ignore_index=True)
dataframe_valid = pd.concat(dataframe_valid_partials, ignore_index=True)

# 'ds' for Dataset
# 'dl' for DataLoader

# get raw data
df_train = dataframe_train_partials[0]  # pick one dataset at a time
df_valid = dataframe_valid_partials[0]
df_test = dataframe_test_partials[0]
# Train
train_ds = DatasetWildfire(df_train, transforms)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
# Valid
valid_ds = DatasetWildfire(df_valid, transforms)
valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False)
# Test
test_ds = DatasetWildfire(df_test, transforms)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)


if __name__ == "__main__":
    # Scratch pad to visualize
    img, label = next(iter(train_dl))
    img = img[0]
    img = de_transforms(img)  # undo normalization
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    # img_pillow = Image.open(os.path.dirname(os.path.abspath(__file__)) /df_train.iloc[0]["ImagePath"])
    # img_pillow.show()
