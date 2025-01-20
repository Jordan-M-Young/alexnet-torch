from app.utils import parquet2images, download_file
import os
def load_data():
    print("Generating Images from Parquet file.")
    parquet2images()


def download_sample_images():
    if not os.path.isdir("./data"):
        os.mkdir("./data")
    
    url = "https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256/resolve/main/data/train-00001-of-00040.parquet?download=true"
    download_file(url)