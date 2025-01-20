"""Run AlexNet Training."""

import numpy as np
from alexnet import AlexNet
from PIL import Image
from utils import get_files_and_labels, get_n_classes


def main():
    """Main training loop."""
    lbl_file = "./data/labels.txt"
    files, labels = get_files_and_labels(lbl_file)
    n_classes = get_n_classes(labels)

    file_dir = "./data/images"
    for file in files:
        file_path = f"{file_dir}/{file}"
        image = Image.open(file_path)
        image.load()
        data = np.asarray(image)

    print(f"Loaded {len(files)} samples")
    print("Array Shape", data.shape)

    _model = AlexNet(n_classes=n_classes)


if __name__ == "__main__":
    main()
