import torch



class AlexNet(torch.nn.module):
    def __init__(self):
        

        self.c1 = torch.conv2d(3, 96, 11,stride=4)
        self.re1 = torch.nn.ReLU()
        torch.nn.LocalResponseNorm(5,0.0001,0.75,2)
        self.mx_pool1 = torch.max_pool2d(kernel_size=3, stride=2)
    
        self.c2 = torch.conv2d(96, 256, 5)
        self.mx_pool1 = torch.max_pool2d(kernel_size=3, stride=2)
        self.re1 = torch.nn.ReLU()

        self.c3 = torch.conv2d(256, 384, 3)
        self.re1 = torch.nn.ReLU()

        self.c4 = torch.conv2d(384, 192, 3)
        self.re1 = torch.nn.ReLU()

        self.c5 = torch.conv2d(192, 256, 3)
        self.mx_pool1 = torch.max_pool2d(kernel_size=3, stride=2)

        self.fc1 = torch.nn.Linear(4096)
        self.re1 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(4096,4096)
        self.re1 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(4096,1000)
        self.re1 = torch.nn.ReLU()

        self.soft = torch.nn.Softmax(dim=1000)
        pass


    def forward(x):
        pass