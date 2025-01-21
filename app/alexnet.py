"""AlexNet class."""

import torch


class AlexNet(torch.nn.Module):
    """AlexNet Class, extended from torch Module."""

    def __init__(self, n_classes=3):
        """Initialize AlexNet Class."""
        super(AlexNet, self).__init__()

        self.c1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4)
        self.re1 = torch.nn.ReLU()
        self.lrn = torch.nn.LocalResponseNorm(5, 0.0001, 0.75, 2)
        self.mx_pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.c2 = torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5))
        self.mx_pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.re2 = torch.nn.ReLU()

        self.c3 = torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3))
        self.re3 = torch.nn.ReLU()

        self.c4 = torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3))
        self.re4 = torch.nn.ReLU()

        self.c5 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3))
        self.re5 = torch.nn.ReLU()
        self.mx_pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = torch.nn.Linear((256 * 2 * 2), 4096)
        self.re6 = torch.nn.ReLU()
        self.dp1 = torch.nn.Dropout(0.5)

        self.fc2 = torch.nn.Linear(4096, 4096)
        self.re7 = torch.nn.ReLU()
        self.dp2 = torch.nn.Dropout(0.5)

        self.fc3 = torch.nn.Linear(4096, n_classes)
        self.re8 = torch.nn.ReLU()

        self.soft = torch.nn.Softmax(dim=n_classes)

    def forward(self, x):
        """Forward pass method."""
        x = self.c1(x)
        x = self.re1(x)
        x = self.lrn(x)
        x = self.mx_pool1(x)
        x = self.c2(x)
        x = self.re2(x)
        x = self.mx_pool2(x)
        x = self.c3(x)
        x = self.re3(x)
        x = self.c4(x)
        x = self.re4(x)
        x = self.c5(x)
        x = self.re5(x)
        x = self.mx_pool3(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.fc1(x)
        x = self.re6(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.re7(x)
        x = self.dp2(x)
        x = self.fc3(x)
        x = self.re8(x)

        return x
