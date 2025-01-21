# Alexnet-Torch
Pytorch Implementation of Alexnet from the [2011 paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

# Data

## Dataset
The original ImageNet 2010 Dataset used in the "ImageNet Classification with Deep Convolutional
Neural Networks" paper is only downloadable from the ImageNet website and one must sign up with an institutional email to gain access to the dataset.0 For this reason I've chosen to work with a specific copy of the dataset hosted on Huggingface by the user [benjamin-paine](https://huggingface.co/benjamin-paine). This dataset is referred to as the [imagenet-1k-256x256](https://huggingface.co/datasets/benjamin-paine/imagenet-1k-256x256/tree/main/data).

This dataset is preferable as the images are already in the 256 px x256 px format the original paper used. 

## Quick Start

To download the whole dataset, I'd suggest using the huggingface/transformers library. To download a fraction of the dataset run the following:

```bash
poetry run download-sample
```

This will download the `train-00000-of-00040.parquet` file from the huggingface repo to this projects data directory. Feel free to extend this basic functionality.

To unpack the downloaded parquet file, run:

```bash
poetry run mk-imgs
```

This will write the byte strings found in the downloaded parquet file to .jpg and write the associated labels to a labels.txt file.

## Full Dataset

The goal of this repo is really only to implement AlexNet in pytoch, if you want to train it on the full ImageNet Dataset, you'll need to write more functionality to pull/parse that data. If there's interest I could do this too.


# Train AlexNet

When you're satisfied with your data setup you can train an instance of AlexNet by running

```
poetry run train
```

This runs the `main` function found in `app/main.py`. Feel free to adjust parameters or make the script a bit more CLI friendly. 