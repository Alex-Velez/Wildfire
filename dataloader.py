"""Contains the required transforms for ImageNet fine-tuning 
and a method to get dataloaders for WildfireDB objects
"""

import os
import yaml

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision.transforms import v2

from wildfiredb import reconstruct_npz

curr_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(curr_dir, "train_config.yaml"), "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

wildfire_transforms = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.Lambda(lambda img: img / 255.0),  # clamp values to [0, 1]
    v2.Resize([224, 224]),
    # v2.Normalize(mean=[0.485, 0.456, 0.406],
    #              std=[0.229, 0.224, 0.225]),  # ImageNet stats
    v2.Normalize(mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5]),  # normalize to [-1, 1]
])
wildfire_npz_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(lambda img: img / 255.0),  # clamp values to [0, 1]
    v2.Resize([224, 224]),
    # v2.Normalize(mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225])
    v2.Normalize(mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5]),  # normalize to [-1, 1]
])
de_wildfire_transforms = v2.Compose([
    # v2.Normalize(
    #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    #     std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    # ),
    v2.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
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
    
    def __init__(self, dataframe, transforms, source: str = ""):
        """
        Args:
            dataframe (pandas.DataFrame): 
                Columns - "ImagePath" | "Image" | "Inferred" | "Class"
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
        self.source = source

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_file = self.dataframe.iloc[idx]["ImagePath"]
        img_file = os.path.dirname(os.path.abspath(__file__)) / img_file
        if img_file.suffix == ".npz":
            image = reconstruct_npz(img_file)
            img_tensor = wildfire_npz_transforms(image)
        else:
            try:
                img_tensor = decode_image(img_file)
            except RuntimeError as e:
                print(f"Error decoding image {img_file}: {e}")
                raise e
        label_str = self.dataframe.iloc[idx]['Class']
        label = self.class_to_idx[label_str]  # convert to integer

        img_transformed = self.transforms(img_tensor)

        return img_transformed, label


class WildfireDataLoaders:
    """Creates train, validm, and test dataloders for a WildFireDB resource and transforms.
    """

    def __init__(self, sources: list, input_transforms: v2.Compose):
        self.sources = sources
        self.config = config
        self.transforms = input_transforms
        self.train_dl, self.valid_dl, self.test_dl = self._get_dataloaders_from_sources(
            self.sources)

    def _get_dataloaders_from_sources(self, sources: list) -> list:
        """Creates Train, Test and Validation DataLoaders for the combined data from `sources`.

        Args:
            sources (list): A list of Wildfire data source instances.
            config (dict): Configuration dictionary containing 'batch_size' and 'num_workers'.
            shuffle (bool, optional): Whether to shuffle the training data. Defaults to False.

        Returns:
            list: A list of tuples containing dataloaders (train, valid, test) for the combined sources.
        """
        train_dfs = []
        valid_dfs = []
        test_dfs = []
        source_names = []

        for source in sources:
            wf_df_train, wf_df_valid, wf_df_test = source.generate_dataframes()
            train_dfs.append(wf_df_train)
            valid_dfs.append(wf_df_valid)
            test_dfs.append(wf_df_test)
            source_names.append(source.DATASET_NAME)

        # Combine dataframes
        combined_train_df = pd.concat(train_dfs, ignore_index=True)
        combined_valid_df = pd.concat(valid_dfs, ignore_index=True)
        combined_test_df = pd.concat(test_dfs, ignore_index=True)

        # Create datasets
        source_names = '_'.join(source_names)
        combined_train_ds = DatasetWildfire(combined_train_df, self.transforms, source=source_names)
        combined_valid_ds = DatasetWildfire(combined_valid_df, self.transforms, source=source_names)
        combined_test_ds = DatasetWildfire(combined_test_df, self.transforms, source=source_names)

        # Create dataloaders
        combined_train_dl = DataLoader(
            combined_train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
        combined_valid_dl = DataLoader(
            combined_valid_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
        combined_test_dl = DataLoader(
            combined_test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])

        return (combined_train_dl, combined_valid_dl, combined_test_dl)
