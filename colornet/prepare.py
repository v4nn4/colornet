import logging
import os
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
torch.manual_seed(1337)


def generate_dataset(path: str) -> Tuple[TensorDataset, TensorDataset]:
    transform_bw = transforms.Compose(
        [
            transforms.Grayscale(),  # Convert to grayscale if needed
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    transform_rgb = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    bw_tensors = []
    rgb_tensors = []
    for filename in tqdm(os.listdir(path)[:100]):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Check for image files
            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            img_bw = transform_bw(img)
            img_rgb = transform_rgb(img)
            bw_tensors.append(img_bw)
            rgb_tensors.append(img_rgb)
    bw_tensors = torch.tensor(np.array(bw_tensors))
    rgb_tensors = torch.tensor(np.array(rgb_tensors))

    # Shuffle
    shuffled_indices = torch.randperm(bw_tensors.size(0))
    bw_tensors = bw_tensors[shuffled_indices]
    rgb_tensors = rgb_tensors[shuffled_indices]

    # Split
    split_index = int(0.8 * len(bw_tensors))
    train_dataset = TensorDataset(bw_tensors[:split_index], rgb_tensors[:split_index])
    test_dataset = TensorDataset(bw_tensors[split_index:], rgb_tensors[split_index:])

    return train_dataset, test_dataset
