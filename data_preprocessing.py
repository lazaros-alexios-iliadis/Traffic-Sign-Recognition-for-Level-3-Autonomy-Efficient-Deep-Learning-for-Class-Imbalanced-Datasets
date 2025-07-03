import pandas as pd
import os
from torchvision import transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split


def load_train_images_from_csv(csv_path, image_folder):
    data = pd.read_csv(csv_path)
    labels = data['ClassId'].values
    image_paths = []

    for _, row in data.iterrows():
        relative_path = row["Path"].replace("Train/", "")
        image_path = os.path.join(image_folder, relative_path)

        if os.path.exists(image_path) and os.path.splitext(image_path)[1].lower() == ".png":
            image_paths.append(image_path)
        else:
            print(f"Image not found or not a PNG file: {image_path}")

    return image_paths, labels


def load_test_images_from_csv(csv_path, image_folder):
    data = pd.read_csv(csv_path)
    labels = data['ClassId'].values
    image_paths = []

    for _, row in data.iterrows():
        relative_path = row["Path"].replace("Test/", "")
        image_path = os.path.join(image_folder, relative_path)

        if os.path.exists(image_path) and os.path.splitext(image_path)[1].lower() == ".png":
            image_paths.append(image_path)
        else:
            print(f"Image not found or not a PNG file: {image_path}")

    return image_paths, labels


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
        transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET, interpolation=InterpolationMode.BILINEAR,
                               fill=(128, 128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return train_transform, test_transform


def split_data(images, labels):
    test_images, val_images, test_labels, val_labels = train_test_split(images, labels, test_size=0.5, random_state=42)
    return test_images, val_images, test_labels, val_labels
