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

        # first group, of conv layers, Section 3. Page 4-5
        self.conv_net = torch.nn.Sequential(
            self.c1,
            self.re1,
            self.lrn,
            self.mx_pool1,
            self.c2,
            self.re2,
            self.mx_pool2,
            self.c3,
            self.re3,
            self.c4,
            self.re4,
            self.c5,
            self.re5,
            self.mx_pool3,
        )

        # fully connected layers group Section 3. Page 4-5
        self.fully_connected = torch.nn.Sequential(
            self.fc1,
            self.re6,
            self.dp1,
            self.fc2,
            self.re7,
            self.dp2,
            self.fc3,
            self.re8,
        )

        # As discussed in Section 5. Page 6
        self.init_biases()
        self.init_weights()

    def forward(self, x):
        """Forward pass method."""
        x = self.conv_net(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.fully_connected(x)

        return x

    def init_biases(self):
        """Initialize biases according to paper methodology."""
        # layers with biases initialized at 0.0 Section 5. Page 6
        self.c1.bias.data.fill_(0.0)
        self.c3.bias.data.fill_(0.0)

        # layers with biasese initialized at 1.0 Section 5. Page 6
        self.c2.bias.data.fill_(1.0)
        self.c4.bias.data.fill_(1.0)
        self.c5.bias.data.fill_(1.0)

        # commented out, if you set these fc layers biases to 1's the model does not learn lol
        # loss explodes till we return nans
        # self.fc1.bias.data.fill_(1.0)
        # self.fc2.bias.data.fill_(1.0)
        # self.fc3.bias.data.fill_(1.0)

    def init_weights(self):
        """Recursively Initialize weight of layers."""
        self.conv_net.apply(initialize_weights)
        self.fully_connected.apply(initialize_weights)


def initialize_weights(m: torch.nn.Module):
    """Initialize weight of layer according to a normal distribution."""
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
