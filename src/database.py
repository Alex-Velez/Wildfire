from pathlib import Path
import pandas
from pandas import DataFrame
import numpy
import torch
import torchvision
import torchvision.transforms.v2
from PIL import Image
from tabulate import tabulate

import paths

PROJECT_NAME = "Wildfire"
COLUMNS = ["RelativePath", "ImageName", "Inferred", "Class"]
CATEGORIES: list[str] = ["fire", "nofire"]
pandas.set_option('display.max_colwidth', 24)

def append_path_or_else(base: Path, paths: list[str]) -> Path:
    for path in paths:
        maybe_path = base / path
        if maybe_path.is_dir():
            return maybe_path
    return base

def generate_dataframe(folder_path: Path) -> DataFrame:
    print(f"[{PROJECT_NAME}]: Generating new dataframe from '{folder_path.relative_to(paths.BASE_DIR)}'", end=" ")
    
    FIRE_DIR: Path = append_path_or_else(folder_path, ["fire", "wildfire"])
    NOFIRE_DIR: Path = append_path_or_else(folder_path, ["nofire", "nowildfire"])
    
    fire_images: list[tuple[Path, str, bool, str]] = [
        (img.relative_to(paths.BASE_DIR), img.name, False, "fire") for img in FIRE_DIR.iterdir() if img.is_file()
    ]
    nofire_images: list[tuple[Path, str, bool, str]] = [
        (img.relative_to(paths.BASE_DIR), img.name, False, "nofire") for img in NOFIRE_DIR.iterdir() if img.is_file()
    ]

    data: list[tuple[Path, str, bool, str]] = fire_images + nofire_images
    dataframe: DataFrame = DataFrame(data, columns=COLUMNS)
    
    print(dataframe.shape)

    return dataframe

def generate_dataframe_npz(folder_path: Path):
    print(f"[{PROJECT_NAME}]: Generating new dataframe from '{folder_path.relative_to(paths.BASE_DIR)}'", end=" ")
    
    data: list[tuple[Path, str, bool, str]] = []
        
    for npz_file_path in folder_path.iterdir():
        with numpy.load(npz_file_path) as npz_data:
            infer_label = "fire" if numpy.any(npz_data["label"] == 1) else "nofire"
            data.append((npz_file_path.relative_to(paths.BASE_DIR), npz_file_path.name, True, infer_label))

    dataframe: DataFrame = DataFrame(data, columns=COLUMNS)
    
    print(dataframe.shape)
    
    return dataframe

class RawDataPaths1():
    def __init__(self) -> None:
        self.DATASET_NAME = "jafar_2023"
        self.DIR: Path = paths.DATA_DIR / self.DATASET_NAME
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

class RawDataPaths2():
    def __init__(self) -> None:
        self.DATASET_NAME = "madafri_2023"
        self.DIR: Path = paths.DATA_DIR / self.DATASET_NAME
        self.TRAIN_DIR: Path = self.DIR / "train"
        self.TEST_DIR: Path = self.DIR / "test"
        self.VALID_DIR: Path = self.DIR / "val"
    
    def generate_dataframes(self) -> list[DataFrame]:
        return [
            generate_dataframe(self.TRAIN_DIR),
            generate_dataframe(self.TEST_DIR),
            generate_dataframe(self.VALID_DIR),
        ]

class RawDataPaths3():
    def __init__(self) -> None:
        self.DATASET_NAME = "aaba_2022"
        self.DIR: Path = paths.DATA_DIR / self.DATASET_NAME
        self.TRAIN_DIR: Path = self.DIR / "train"
        self.TEST_DIR: Path = self.DIR / "test"
        self.VALID_DIR: Path = self.DIR / "valid"
    
    def generate_dataframes(self) -> list[DataFrame]:
        return [
            generate_dataframe(self.TRAIN_DIR),
            generate_dataframe(self.TEST_DIR),
            generate_dataframe(self.VALID_DIR),
        ]

class RawDataPaths4():
    def __init__(self) -> None:
        self.DATASET_NAME = "xu_2024"
        self.DIR: Path = paths.DATA_DIR / self.DATASET_NAME
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

def generate_database(
    raw_data_paths: list[RawDataPaths1 | RawDataPaths2 | RawDataPaths3 | RawDataPaths4]
) -> tuple[DataFrame, DataFrame, DataFrame]:
    dataframe_partials: list[list[DataFrame]] = [data_path.generate_dataframes() for data_path in raw_data_paths]

    dataframe_train_partials: list[DataFrame] = [dataframe_partial[0] for dataframe_partial in dataframe_partials]
    dataframe_test_partials: list[DataFrame] = [dataframe_partial[1] for dataframe_partial in dataframe_partials]
    dataframe_valid_partials: list[DataFrame] = [dataframe_partial[2] for dataframe_partial in dataframe_partials]

    dataframe_train: DataFrame = pandas.concat(dataframe_train_partials, ignore_index=True)
    dataframe_test: DataFrame = pandas.concat(dataframe_test_partials, ignore_index=True)
    dataframe_valid: DataFrame = pandas.concat(dataframe_valid_partials, ignore_index=True)
    
    return (dataframe_train, dataframe_test, dataframe_valid)
    
if __name__ == "__main__":
    (dataframe_train, dataframe_test, dataframe_valid) = generate_database(
        [
            RawDataPaths1(),
            RawDataPaths2(),
            RawDataPaths3(),
            RawDataPaths4(),
        ]
    )
    
    train_count: int = dataframe_train.shape[0]
    test_count: int = dataframe_test.shape[0]
    valid_count: int = dataframe_valid.shape[0]
    total_count: int = train_count + test_count + valid_count
    
    train_percent: float = round((train_count/total_count) * 100, 2)
    test_percent: float = round((test_count/total_count) * 100, 2)
    valid_percent: float = round((valid_count/total_count) * 100, 2)
    
    info_table = tabulate(
        [
            ["Images", "Count", "%"],
            ["Training", f"{train_count}/{total_count}", f"{train_percent}%"],
            ["Testing", f"{test_count}/{total_count}", f"{test_percent}%"],
            ["Validation", f"{valid_count}/{total_count}", f"{valid_percent}%"],
        ],
        headers="firstrow",
        tablefmt="simple_grid",
        numalign="center",
        stralign="center",
    )
    print(info_table)
    
    print("Example DataFrame:")
    print(dataframe_train)