"""Run AlexNet Training."""

from torch import optim
from torch.utils.data import DataLoader, random_split

from app.alexnet import AlexNet
from app.data import ImageDataset, load_data
from app.training import eval, train


def main():
    """Main training loop."""
    # Model/Optimizer parameters as discussed in paper.
    BATCH_SIZE = 128
    TEST_FRACTION = 0.2
    INIT_LR = 0.01
    WEIGHT_DECAY = 0.0005
    EPOCHS = 10
    LR_REDUCE_FACTOR = 0.1
    MOMENTUM = 0.9

    # load dataset.
    data = load_data()
    images = data["images"]
    labels = data["labels"]
    n_classes = data["n_classes"]
    dataset = ImageDataset(images=images, labels=labels, n_classes=n_classes)

    # split dataset
    size = len(dataset)
    train_size = int((1 - TEST_FRACTION) * size)
    test_size = size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # initialize model, optimizer, and lr scheduler
    model = AlexNet(n_classes=n_classes)

    # you'll probably want to adjust this... Adam with a lower lr works much better lol
    optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=LR_REDUCE_FACTOR, patience=30)

    # run training/testing
    for epoch in range(EPOCHS):
        tr_epoch_loss = train(train_dataloader, model, optimizer)
        ev_epoch_loss = eval(test_dataloader, model)
        print(
            f"Epoch {epoch} Loss: ",
            tr_epoch_loss / len(train_dataset),
            f" Epoch {epoch} Test Loss: ",
            ev_epoch_loss / len(test_dataset),
        )
        scheduler.step(ev_epoch_loss)


if __name__ == "__main__":
    main()
