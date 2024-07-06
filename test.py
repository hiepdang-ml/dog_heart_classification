import os
from typing import List, Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

transformation = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

data_root = r'data/labeled'
dataset = ImageFolder(root=data_root, transform=transformation)
