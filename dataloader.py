import pandas
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision.transforms import v2

# from wildfire import dataframe_train, dataframe_test

# Placeholder
dataframe_train = pandas.DataFrame(
    # torch.rand(2000, 3),
    # ["jafar_2023",
    #  r"Wildfire\data\jafar_2023\Classification\test\fire\fire (3613).png",
    #  "fire"] * 2000,
    [{'Dataset': "jafar_2023",
     'Image': r"Wildfire\data\jafar_2023\Classification\test\fire\fire (3613).png",
     'Class': "fire"},
     {'Dataset': "madafri_2023",
     'Image': r"Wildfire\data\madafri_2023\Classification\test\fire\fire (1001).jpg",
     'Class': "fire"}] * 1000,
    index=range(2000)
    # columns=['Dataset', 'Image', 'Class']
)
dataframe_test = pandas.DataFrame(
    torch.rand(2000, 3),
    columns=['Dataset', 'Image', 'Class']
)
dataframe_valid = pandas.DataFrame(
    torch.rand(2000, 3),
    columns=['Dataset', 'Image', 'Class']
)

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
    def __init__(self, dataframe: pandas.DataFrame, transforms):
        """
        Args:
            dataframe (pandas.DataFrame): Columns - "Dataset" | "Image" | "Class"
                                          "Dataset" - Name of dataset
                                          "Image" - Path to image file
                                          "Class" - "fire" or "nofire"
            transforms (torchvision.transforms): image preprocessing steps defined in torchvision.transforms
        """
        self.dataframe = dataframe
        self.transforms = transforms

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_file = self.dataframe.iloc[idx]["Image"]
        img_tensor = decode_image(img_file)
        label = self.dataframe.iloc[idx]['Class']

        img_transformed = self.transforms(img_tensor)

        return img_transformed, label

# 'ds' for Dataset
# 'dl' for DataLoader

# Train
jafar_2023_train_ds = DatasetWildfire(
    dataframe_train[dataframe_train['Dataset'] == 'jafar_2023'].reindex(),
    transforms
)
madafri_2023_train_ds = DatasetWildfire(
    dataframe_train[dataframe_train['Dataset'] == 'madafri_2023'].reindex(),
    transforms
)

jafar_2023_train_dl = DataLoader(
    jafar_2023_train_ds, batch_size=1, shuffle=True)
madafri_2023_train_dl = DataLoader(
    madafri_2023_train_ds, batch_size=1, shuffle=True)

# Valid
jafar_2023_valid_ds = DatasetWildfire(
    dataframe_valid[dataframe_valid['Dataset'] == 'jafar_2023'].reindex(),
    transforms
)
madafri_2023_valid_ds = DatasetWildfire(
    dataframe_valid[dataframe_valid['Dataset'] == 'madafri_2023'].reindex(),
    transforms
)

# jafar_2023_valid_dl = DataLoader(
#     jafar_2023_valid_ds, batch_size=1, shuffle=False)
# madafri_2023_valid_dl = DataLoader(
#     madafri_2023_valid_ds, batch_size=1, shuffle=False)

# Test
jafar_2023_test_ds = DatasetWildfire(
    dataframe_test[dataframe_test['Dataset'] == 'jafar_2023'].reindex(),
    transforms
)
madafri_2023_test_ds = DatasetWildfire(
    dataframe_test[dataframe_test['Dataset'] == 'madafri_2023'].reindex(),
    transforms
)

# jafar_2023_test_dl = DataLoader(
#     jafar_2023_test_ds, batch_size=1, shuffle=False)
# madafri_2023_test_dl = DataLoader(
#     madafri_2023_test_ds, batch_size=1, shuffle=False)



# Scratch pad to visualize
from matplotlib import pyplot as plt
from PIL import Image
img, label = next(iter(jafar_2023_train_dl))

img = de_transforms(img[0])  # undo normalization
plt.imshow(img.permute(1, 2, 0))
plt.show()
img_pillow = Image.open(dataframe_train.iloc[0]["Image"])
img_pillow.show()
