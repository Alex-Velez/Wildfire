from pathlib import Path
import pandas
from pandas import DataFrame
import numpy
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.v2
from torchvision.transforms.v2 import Compose, ToDtype, Lambda, Resize, Normalize
from PIL import Image
import yaml

import paths
import dataset


wildfire_transforms = Compose([
    ToDtype(torch.float32),
    Lambda(lambda img: img / 255.0),  # clamp values to [0, 1]
    Resize([224, 224]),
    Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),  # ImageNet stats
])

de_wildfire_transforms = Compose([
    Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    ),
    Lambda(lambda img: img * 255.0),
    ToDtype(torch.uint8),
])

class WildfireDataLoaders:
    """Creates train, validm, and test dataloders for a WildFireDB resource and transforms.
    """

    def __init__(self, sources: list, input_transforms: Compose):
        self.sources = sources
        
        with open(paths.CONFIG_DIR / "train_config.yaml", "r", encoding='utf-8') as config_file:
            self.config = yaml.safe_load(config_file)
            
        self.transforms = input_transforms
        self.train_dl, self.valid_dl, self.test_dl = self._get_dataloaders_from_sources(self.sources)

    def _get_dataloaders_from_sources(self, sources: list) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Creates Train, Test and Validation DataLoaders for the combined data from `sources`.

        Args:
            sources (list): A list of Wildfire data source instances.
            config (dict): Configuration dictionary containing 'batch_size' and 'num_workers'.
            shuffle (bool, optional): Whether to shuffle the training data. Defaults to False.

        Returns:
            list: A list of tuples containing dataloaders (train, valid, test) for the combined sources.
        """
        train_dataframes = []
        valid_dataframes = []
        test_dataframes = []
        source_names = []

        for source in sources:
            wf_df_train, wf_df_valid, wf_df_test = source.generate_dataframes()
            train_dataframes.append(wf_df_train)
            valid_dataframes.append(wf_df_valid)
            test_dataframes.append(wf_df_test)
            source_names.append(source.DATASET_NAME)

        # Combine dataframes
        combined_train_df = pandas.concat(train_dataframes, ignore_index=True)
        combined_valid_df = pandas.concat(valid_dataframes, ignore_index=True)
        combined_test_df = pandas.concat(test_dataframes, ignore_index=True)

        # Create datasets
        source_names = '_'.join(source_names)
        combined_train_ds = dataset.WildfireDataset(combined_train_df, self.transforms, source=source_names)
        combined_valid_ds = dataset.WildfireDataset(combined_valid_df, self.transforms, source=source_names)
        combined_test_ds = dataset.WildfireDataset(combined_test_df, self.transforms, source=source_names)

        # Create dataloaders
        combined_train_dl = DataLoader(
            combined_train_ds, batch_size=self.config["batch_size"], shuffle=True, num_workers=self.config["num_workers"])
        combined_valid_dl = DataLoader(
            combined_valid_ds, batch_size=self.config["batch_size"], shuffle=False, num_workers=self.config["num_workers"])
        combined_test_dl = DataLoader(
            combined_test_ds, batch_size=self.config["batch_size"], shuffle=False, num_workers=self.config["num_workers"])

        return (combined_train_dl, combined_valid_dl, combined_test_dl)
