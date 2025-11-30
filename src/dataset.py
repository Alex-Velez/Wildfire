from pathlib import Path
from pandas import DataFrame
import numpy
import torch
import torchvision
import torchvision.transforms.v2
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from PIL import Image

import paths

# normalize per band to [0, 255]
def norm(x) -> int:
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
    return None

class WildfireDataset(torch.utils.data.Dataset):
    """Transforms Wildfire dataframe into a PyTorch Dataset.
    Label "fire" is mapped to 1, "nofire" to 0.

    Args:
        Dataset (_type_): _description_
    """
    
    def __init__(self, dataframe, transforms: Compose, source: str = ""):
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
        self.dataframe: DataFrame = dataframe
        self.transforms: Compose = transforms
        self.class_encoding: dict[str, int] = {"fire": 1, "nofire": 0} # label numerical-boolean encoding
        self.source = source

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        image_relative_path: Path = Path(self.dataframe.iloc[idx]["RelativePath"])
        image_absolute_path: Path = paths.BASE_DIR / image_relative_path
        image_tensor: torch.Tensor
        
        
        if image_absolute_path.suffix == ".npz":
            image = reconstruct_npz(image_absolute_path)
            image_tensor = Compose(
                [
                    ToImage(),
                    ToDtype(torch.float32, scale=True)
                ]
            )(image)
        else:
            try:
                image_tensor = torchvision.io.decode_image(str(image_absolute_path))
            except RuntimeError as e:
                print(f"Error decoding image {image_absolute_path}: {e}")
                raise e
        
        label_str = self.dataframe.iloc[idx]["Class"]
        label_encoded = self.class_encoding[label_str] # encode label

        image_transformed = self.transforms(image_tensor)

        return image_transformed, label_encoded
