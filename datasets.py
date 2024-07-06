import os
from typing import List, Tuple

from PIL import Image

import torch
import torch.utils
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class DogHeartLabeledDataset(ImageFolder):

    #extend
    def __init__(self, data_root: str) -> None:
        self.transformation = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor(),
        ])
        super().__init__(root=data_root, transform=self.transformation)
        self.data_root: str = data_root

        self.filepaths: List[str] = [path for path, _ in self.samples]
        self.filenames: List[str] = [path.split('/')[-1] for path in self.filepaths]
        self.labels: List[int] = [label for _, label in self.samples]

    #extend
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        tensor: torch.Tensor; label: int
        tensor, label = super().__getitem__(idx)
        filename: str = self.filenames[idx]
        return tensor, label, filename

class DogHearUnlabeledDataset(Dataset):

    def __init__(self, data_root: str) -> None:
        self.data_root: str = data_root
        self.transformation = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor(),
        ])
        self.filenames: List[str] = os.listdir(self.data_root)
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        filename: str = self.filenames[idx]
        image: Image = Image.open(os.path.join(self.data_root, filename))
        tensor: torch.Tensor = self.transformation(image)
        return tensor, filename


if __name__ == '__main__':

    # # P1.Q1

    # train_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Train')
    # valid_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Valid')
    # test_dataset = DogHearUnlabeledDataset(data_root='Test')

    # train_dataloader: DataLoader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    # valid_dataloader: DataLoader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False)
    # test_dataloader: DataLoader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    self = DogHeartLabeledDataset(data_root='Dog_heart/Train')







