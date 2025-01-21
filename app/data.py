"""Classes and functions for data handling."""

import numpy as np
import utils as ut
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Custom ImageDataset class, extends basic Dataset class."""

    def __init__(self, images, labels, n_classes):
        """Initialize image dataset."""
        self.images = images
        self.n_classes = n_classes
        self.labels = encode_labels(n_labels=n_classes, labels=labels)

    def __getitem__(self, idx):
        """Get a feature and target from dataset."""
        image = self.images[idx]
        target = self.labels[idx]

        return image, target

    def __len__(self):
        """Get len of the dataset."""
        return len(self.images)


def encode_labels(n_labels, labels) -> list[list]:
    """Encode Integer Labels into Sparse Array Targets."""
    encoded = [0.0] * n_labels

    encoded_labels = [encoded] * len(labels)

    for idx, label in enumerate(labels):
        i_label = int(label)
        encoded_labels[idx][i_label] = 1.0

    return encoded_labels


def load_data() -> dict:
    """Loads data, kindof a bad function in this state."""
    lbl_file = "./data/labels.txt"
    files, labels = ut.get_files_and_labels(lbl_file)
    n_classes = ut.get_n_classes(labels)

    file_dir = "./data/images"
    images = []
    for _, file in enumerate(files):
        file_path = f"{file_dir}/{file}"
        image = Image.open(file_path).convert("RGB")
        image.load()
        data = np.asarray(image, dtype="float32")
        data = np.moveaxis(data, -1, 0)
        images.append(data)
    print(f"Loaded {len(files)} samples")
    print("Array Shape", data.shape)

    return {"images": images, "labels": labels, "n_classes": n_classes}
