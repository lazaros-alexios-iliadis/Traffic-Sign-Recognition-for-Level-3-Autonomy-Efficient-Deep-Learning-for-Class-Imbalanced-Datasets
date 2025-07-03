import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


class TrafficSignDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error reading image {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            pil_image = self.transform(pil_image)

        return pil_image, label

