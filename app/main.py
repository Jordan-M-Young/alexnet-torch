"""Run AlexNet Training."""

import numpy as np
from PIL import Image
from torch import optim, tensor
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, random_split

from app.alexnet import AlexNet
from app.data import ImageDataset
from app.utils import get_files_and_labels, get_n_classes


def main():
    """Main training loop."""
    lbl_file = "./data/labels.txt"
    files, labels = get_files_and_labels(lbl_file)
    n_classes = get_n_classes(labels)

    file_dir = "./data/images"
    images = []
    for _idx, file in enumerate(files):
        file_path = f"{file_dir}/{file}"
        image = Image.open(file_path).convert("RGB")
        image.load()
        data = np.asarray(image, dtype="float32")
        data = np.moveaxis(data, -1, 0)
        images.append(data)
    print(f"Loaded {len(files)} samples")
    print("Array Shape", data.shape)

    dataset = ImageDataset(images=images, labels=labels, n_classes=n_classes)

    batch_size = 128

    test_frac = 0.2
    size = len(dataset)
    train_size = int((1 - test_frac) * size)
    test_size = size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    _test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = AlexNet(n_classes=n_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1)
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for _, batch in enumerate(train_dataloader):
            inputs, labels = batch
            output = model(inputs)
            labels = np.array(labels)
            labels = labels.reshape(len(inputs), n_classes)
            labels = tensor(labels)

            loss = cross_entropy(output, tensor(labels))
            loss_val = loss.detach().item()
            epoch_loss += loss_val
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss: ", epoch_loss / len(train_dataloader))
        scheduler.step(epoch_loss)


if __name__ == "__main__":
    main()
