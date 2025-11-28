

import sys

# required 3.12 version
# if sys.version_info[3][:2] < (3, 12): 
#     raise Exception(f"Python 3.12 is required (Current is {[sys.version_info[i] for i in range(3)]})")

# import time
# import random
# import matplotlib.pyplot as plt
# import seaborn as sns
# import cv2
# import imutils
# import PIL
from PIL import Image
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

import torch
import os
from pathlib import Path
import pandas
from pandas import DataFrame
import numpy

PROGRAM_NAME: str = "Wildfire"
COLUMNS = ["ImagePath", "Image", "Inferred", "Class"]
CATEGORIES: list[str] = ["fire", "nofire"]
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
# __file__ = os.getcwd()
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
pandas.set_option('display.max_colwidth', 24)


def append_path_or_else(base: Path, paths: list[str]) -> Path:
    for path in paths:
        maybe_path = base / path
        if maybe_path.is_dir():
            return maybe_path
    return base


def generate_dataframe(folder_path: Path) -> DataFrame:
    print(f"[{PROGRAM_NAME}]: Generating new dataframe from '{folder_path.relative_to(BASE_DIR)}'", end=" ")
    
    FIRE_DIR: Path = append_path_or_else(folder_path, ["fire", "wildfire"])
    NOFIRE_DIR: Path = append_path_or_else(folder_path, ["nofire", "nowildfire"])
    
    fire_images: list[tuple[Path, str, bool, str]] = [
        (img.relative_to(BASE_DIR), img.name, False, "fire") for img in FIRE_DIR.iterdir() if img.is_file()
    ]
    nofire_images: list[tuple[Path, str, bool, str]] = [
        (img.relative_to(BASE_DIR), img.name, False, "nofire") for img in NOFIRE_DIR.iterdir() if img.is_file()
    ]

    data: list[tuple[Path, str, bool, str]] = fire_images + nofire_images
    dataframe: DataFrame = DataFrame(data, columns=COLUMNS)
    
    print(dataframe.shape)

    return dataframe


def generate_dataframe_npz(folder_path: Path):
    print(f"[{PROGRAM_NAME}]: Generating new dataframe from '{folder_path.relative_to(BASE_DIR)}'", end=" ")
    
    data: list[tuple[Path, str, bool, str]] = []
        
    for npz_file_path in folder_path.iterdir():
        with numpy.load(npz_file_path) as npz_data:
            infer_label = "fire" if numpy.any(npz_data["label"] == 1) else "nofire"
            data.append((npz_file_path.relative_to(BASE_DIR), npz_file_path.name, True, infer_label))

    dataframe: DataFrame = DataFrame(data, columns=COLUMNS)
    
    print(dataframe.shape)
    
    return dataframe



class WildFireData1():
    def __init__(self) -> None:
        self.DATASET_NAME = "jafar_2023"
        self.DIR: Path = DATA_DIR / self.DATASET_NAME
        self.CLASS_DIR: Path = self.DIR / "Classification"
        self.CLASS_TRAIN_DIR: Path = self.CLASS_DIR / "train"
        self.CLASS_TEST_DIR: Path = self.CLASS_DIR / "test"
        self.CLASS_VALID_DIR: Path = self.CLASS_DIR / "valid"
        self.DETECTION_DIR: Path = self.DIR / "Detection"
        self.DETECTION_TRAIN_DIR: Path = self.DETECTION_DIR / "train"
        self.DETECTION_TEST_DIR: Path = self.DETECTION_DIR / "test"
        self.DETECTION_VALID_DIR: Path = self.DETECTION_DIR / "valid"
    
    def generate_dataframes(self) -> list[DataFrame]:
        return [
            generate_dataframe(self.CLASS_TRAIN_DIR),
            generate_dataframe(self.CLASS_TEST_DIR),
            generate_dataframe(self.CLASS_VALID_DIR),
        ]

class WildFireData2():
    def __init__(self) -> None:
        self.DATASET_NAME = "madafri_2023"
        self.DIR: Path = DATA_DIR / self.DATASET_NAME
        self.TRAIN_DIR: Path = self.DIR / "train"
        self.TEST_DIR: Path = self.DIR / "test"
        self.VALID_DIR: Path = self.DIR / "val"
    
    def generate_dataframes(self) -> list[DataFrame]:
        return [
            generate_dataframe(self.TRAIN_DIR),
            generate_dataframe(self.TEST_DIR),
            generate_dataframe(self.VALID_DIR),
        ]

class WildFireData3():
    def __init__(self) -> None:
        self.DATASET_NAME = "aaba_2022"
        self.DIR: Path = DATA_DIR / self.DATASET_NAME
        self.TRAIN_DIR: Path = self.DIR / "train"
        self.TEST_DIR: Path = self.DIR / "test"
        self.VALID_DIR: Path = self.DIR / "valid"
    
    def generate_dataframes(self) -> list[DataFrame]:
        return [
            generate_dataframe(self.TRAIN_DIR),
            generate_dataframe(self.TEST_DIR),
            generate_dataframe(self.VALID_DIR),
        ]

class WildFireData4():
    def __init__(self) -> None:
        self.DATASET_NAME = "xu_2024"
        self.DIR: Path = DATA_DIR / self.DATASET_NAME
        self.SCENE_1_DIR: Path = self.DIR / "scene1"
        self.SCENE_2_DIR: Path = self.DIR / "scene2"
        self.SCENE_3_DIR: Path = self.DIR / "scene3"
        self.SCENE_4_DIR: Path = self.DIR / "scene4"
    
    def generate_dataframes(self) -> list[DataFrame]:
        dataframe_100 = pandas.concat(
            [
                generate_dataframe_npz(self.SCENE_1_DIR),
                generate_dataframe_npz(self.SCENE_2_DIR),
                generate_dataframe_npz(self.SCENE_3_DIR),
                generate_dataframe_npz(self.SCENE_4_DIR),
            ],
            ignore_index = True
        )
        
        random_num = 69 # random state for reproducibility
        dataframe_70 = dataframe_100.sample(frac=0.70, random_state=random_num) 
        dataframe_30 = dataframe_100.drop(dataframe_70.index)
        dataframe_15_1 = dataframe_30.sample(frac=0.50, random_state=random_num)
        dataframe_15_2 = dataframe_30.drop(dataframe_15_1.index)
             
        return [
            dataframe_70,
            dataframe_15_1,
            dataframe_15_2,
        ]


# normalize per band to [0, 255]
def norm(x):
    x = x.astype(numpy.float32)
    x -= x.min()
    x /= (x.max() - x.min() + 1e-6)
    return (x * 255).astype(numpy.uint8)

def reconstruct_npz(file_path: Path) -> Image.Image | None:
    if file_path.is_file():
        with numpy.load(file_path) as npz_data: # image, aerosol, label
            image_label_data = npz_data["image"]
            # select RGB bands (B4, B3, B2)
            red = image_label_data[3]
            green = image_label_data[2]
            blue = image_label_data[1]                
            rgb = numpy.stack([red, green, blue], axis=-1)
            rgb = numpy.dstack([norm(red), norm(green), norm(blue)])
            return Image.fromarray(rgb)

if __name__ == "__main__":
    wildfire_data_1: WildFireData1 = WildFireData1()
    wildfire_data_2: WildFireData2 = WildFireData2()
    wildfire_data_3: WildFireData3 = WildFireData3()
    wildfire_data_4: WildFireData4 = WildFireData4()

    dataframe_partials: list[list[DataFrame]] = [
        wildfire_data_1.generate_dataframes(),
        # wildfire_data_2.generate_dataframes(),
        # wildfire_data_3.generate_dataframes(),
        # wildfire_data_4.generate_dataframes(),
    ]

    dataframe_train_partials: list[DataFrame] = [dataframe_partial[0] for dataframe_partial in dataframe_partials]
    dataframe_test_partials: list[DataFrame] = [dataframe_partial[1] for dataframe_partial in dataframe_partials]
    dataframe_valid_partials: list[DataFrame] = [dataframe_partial[2] for dataframe_partial in dataframe_partials]


    dataframe_train: DataFrame = pandas.concat(dataframe_train_partials, ignore_index=True)
    dataframe_test: DataFrame = pandas.concat(dataframe_test_partials, ignore_index=True)
    dataframe_valid: DataFrame = pandas.concat(dataframe_valid_partials, ignore_index=True)


    train_count: int = dataframe_train.shape[0]
    test_count: int = dataframe_test.shape[0]
    valid_count: int = dataframe_valid.shape[0]
    total_count: int = train_count + test_count + valid_count
    train_percent: float = round(train_count/total_count, 2)
    test_percent: float = round(test_count/total_count, 2)
    valid_percent: float = round(valid_count/total_count, 2)
    # info_table = tabulate(
    #     [
    #         ["Images", "Count", "%"],
    #         ["Training", f"{train_count}/{total_count}", f"{train_percent}%"],
    #         ["Testing", f"{test_count}/{total_count}", f"{test_percent}%"],
    #         ["Validation", f"{valid_count}/{total_count}", f"{valid_percent}%"],
    #     ],
    #     headers="firstrow",
    #     tablefmt="simple_grid",
    #     numalign="center",
    #     stralign="center",
    # )

    # print()
    # print(info_table)




