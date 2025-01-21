"""Utility Functions for Training AlexNet."""

import io
import os
from typing import Tuple

import pandas as pd
import requests
from PIL import Image


def parquet2images():
    """Opens downloaded parquet file and saves image/label data to disk."""
    if not os.path.isdir("./data"):
        os.mkdir("./data")

    data_path = "./data/images"
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    df = pd.read_parquet("./data/train-00000-of-00040.parquet")

    imgs = df["image"].to_list()
    labels = df["label"].to_list()
    used_labels = []
    for idx in range(len(imgs)):
        if idx % 10 == 0:
            img_bytes = imgs[idx]["bytes"]
            label = labels[idx]
            image = Image.open(io.BytesIO(img_bytes))
            save_path = f"{data_path}/{idx}.jpg"
            image.save(save_path)
            used_label = f"{idx}-{label}\n"
            used_labels.append(used_label)
    with open("./data/labels.txt", "w") as tfile:
        tfile.writelines(used_labels)


def open_labels_file(filepath) -> list[list]:
    """Opens label file and returns data."""
    with open(filepath, "r") as tfile:
        data = tfile.readlines()

    return data


def get_files_and_labels(filepath) -> Tuple[list, list]:
    """Gets filenames and labels from label.txt file."""
    data = open_labels_file(filepath)
    labels = []
    file_names = []

    for line in data:
        line = line.replace("\n", "").strip()
        file, label = line.split("-")
        labels.append(label)
        file_names.append(f"{file}.jpg")

    return file_names, labels


def get_n_classes(labels: list) -> int:
    """Get number of distinct classes from label list."""
    n_classes = 0
    class_set = set()
    for label in labels:
        if label in class_set:
            continue

        class_set.add(label)
        n_classes += 1
    return n_classes


def download_file(url):
    """Download parquet file from url."""
    file = url.split("/")[-1].split("?")[0]
    save_path = f"./data/{file}"

    try:
        resp = requests.get(url)
        with open(save_path, "wb") as file:
            file.write(resp.content)
    except Exception as e:
        print("Download Failed!\n", e)
