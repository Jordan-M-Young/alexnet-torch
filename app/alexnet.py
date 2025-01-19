import torch



class AlexNet(torch.nn.module):
    def __init__(self):
        

        self.c1 = torch.conv2d(3, 96, 11,stride=4)
        self.re1 = torch.nn.ReLU()
        self.lrn = torch.nn.LocalResponseNorm(5,0.0001,0.75,2)
        self.mx_pool1 = torch.max_pool2d(kernel_size=3, stride=2)
    
        self.c2 = torch.conv2d(96, 256, 5)
        self.mx_pool2 = torch.max_pool2d(kernel_size=3, stride=2)
        self.re2 = torch.nn.ReLU()

        self.c3 = torch.conv2d(256, 384, 3)
        self.re3 = torch.nn.ReLU()

        self.c4 = torch.conv2d(384, 192, 3)
        self.re4 = torch.nn.ReLU()

        self.c5 = torch.conv2d(192, 256, 3)
        self.re5 = torch.nn.ReLU()

        self.mx_pool3 = torch.max_pool2d(kernel_size=3, stride=2)

        self.fc1 = torch.nn.Linear(4096,4096)
        self.re6 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(4096,4096)
        self.re7 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(4096,1000)
        self.re8 = torch.nn.ReLU()

        self.soft = torch.nn.Softmax(dim=1000)
        pass


    def forward(self, x):
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
        x = self.fc1(x)
        x = self.re6(x)
        x = self.fc2(x)
        x = self.re7(x)
        x = self.fc3(x)
        x = self.re8(x)
        x = self.soft(x)
        
        return x
        