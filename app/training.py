"""Traininng + Testing functions."""

import numpy as np
from torch import optim, tensor
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from app.alexnet import AlexNet


def train(data: DataLoader, model: AlexNet, optimizer: optim.Optimizer) -> float:
    """Model Training Loop."""
    model.train()
    tr_epoch_loss = 0
    for _, batch in enumerate(data):
        inputs, labels = batch
        output = model(inputs)
        labels = np.array(labels)
        labels = labels.reshape(len(inputs), output.shape[1])
        labels = tensor(labels)

        loss = cross_entropy(output, labels)
        loss_val = loss.detach().item()
        tr_epoch_loss += loss_val
        loss.backward()
        optimizer.step()
    return tr_epoch_loss


def eval(data: DataLoader, model: AlexNet) -> float:
    """Model Testing Loop."""
    model.eval()
    ev_epoch_loss = 0
    for _, batch in enumerate(data):
        inputs, labels = batch
        output = model(inputs)
        labels = np.array(labels)
        labels = labels.reshape(len(inputs), output.shape[1])
        labels = tensor(labels)

        loss = cross_entropy(output, labels)
        loss_val = loss.detach().item()
        ev_epoch_loss += loss_val

    return ev_epoch_loss
