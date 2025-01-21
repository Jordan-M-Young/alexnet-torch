"""Classes and functions for data handling."""

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


# def get_dataloaders(dataset: Dataset, test_size=0.2) -> Tuple(DataLoader, DataLoader):
#     size = len(dataset)
#     train_size = int((1-test_size) * size)
#     test_size = size - train_size

#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


#     train_dataloader = DataLoader(train_dataset,bat)
