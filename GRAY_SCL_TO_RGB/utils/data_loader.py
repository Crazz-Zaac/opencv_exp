from pydantic import BaseModel
from pathlib import Path
from typing import List, Tuple, Callable, Optional
from enum import Enum
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import cv2


class ImageChannel(Enum):
    GRAY = "gray"
    RGB = "rgb"
    HSV = "hsv"


class DataType(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


# force the user to provide the data_path, batch_size, num_workers, pin_memory, and input_size
class DataConfig(BaseModel):
    data_path: Path
    batch_size: int
    num_workers: int
    pin_memory: bool
    input_size: Tuple[int, int]
    train_size: float = 0.8
    seed: int = 42
    shuffle: bool = True
    image_channels: ImageChannel = ImageChannel.RGB
    image_extensions: List[str] = ["jpg", "jpeg", "png"]
    max_data: int = -1 # -1 means all data

    class Config:
        arbitrary_types_allowed = True


class ImageDataset(Dataset):
    def __init__(
        self,
        config: DataConfig,
        transforms: Optional[Callable] = None,
        data_type: DataType = DataType.TRAIN,
        return_type: str = "image",
    ):
        self.config = config
        self.data_path = config.data_path
        self.input_size = config.input_size
        self.image_channels = config.image_channels
        self.image_files = self.get_image_files()
        self.random_state = np.random.RandomState(config.seed)
        if self.config.shuffle:
            self.random_state.shuffle(self.image_files)
        self.transforms = transforms
        self.train_images, self.test_images = train_test_split(
            self.image_files,
            train_size=config.train_size,
            random_state=self.random_state,
        )
        self.data = (
            self.train_images if data_type == DataType.TRAIN else self.test_images
        )

    def get_image_files(self):
        image_files = []
        for ext in self.config.image_extensions:
            image_files.extend(list(self.data_path.rglob(f"*.{ext}")))
        return image_files

    # def item_getter(self, idx, getter_obj: object):

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = cv2.imread(str(image_path))

        if self.image_channels == ImageChannel.GRAY:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.image_channels == ImageChannel.HSV:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, self.input_size)
        if self.transforms:
            image = self.transforms(image)
        
        # if return_type is tensor, convert the image to tensor
        if self.return_type == "tensor":
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            image = torch.tensor(image, dtype=torch.float32)
        
        return image


if __name__ == "__main__":
    config = DataConfig(
        data_path=Path(
            "/home/blackphoenix/projects/opencv_exp/GRAY_SCL_TO_RGB/coco2017/train2017"
        ),
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        input_size=(256, 256),
        train_test_split=[0.8, 0.2],
        image_channels=ImageChannel.RGB,
        image_extensions=["jpg", "jpeg", "png"],
    )
    dataset = ImageDataset(config)
    print(dataset[1])
