import torch.nn as nn
from collections import OrderedDict

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(OrderedDict([
            ('C1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1, padding=0)),
            ('Sm', nn.Sigmoid()),
            ('S2', nn.AvgPool2d(kernel_size=(2, 2), stride = 2, padding=0)),
        ]))

        self.block2 = nn.Sequential(OrderedDict([
            ('C3', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=1, padding=0)),
            ('Sm', nn.Sigmoid()),
            ('S4', nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0)),
        ]))

        self.block3 = nn.Sequential(OrderedDict([
            ('Ft', nn.Flatten()),
            ('C5', nn.Linear(in_features=16*5*5, out_features=120)),
            ('Sm', nn.Sigmoid()),
            ('F6', nn.Linear(in_features=120, out_features=84)),
            ('Sm', nn.Sigmoid()),
            ('Output', nn.Linear(in_features=84, out_features=10))
        ]))
       

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

def get_model(model_name = 'LeNet'):
    if model_name == 'LeNet':
        return LeNet()
