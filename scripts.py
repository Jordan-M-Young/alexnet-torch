"""Poetry Package Scripts."""

import os

from app.utils import download_file, parquet2images


def load_data():
    """Load data script."""
    print("Generating Images from Parquet file.")
    parquet2images()


def download_sample_images():
    """Download sample data script."""
    if not os.path.isdir("./data"):
        os.mkdir("./data")

    url = "https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256/resolve/main/data/train-00001-of-00040.parquet?download=true"
    download_file(url)
