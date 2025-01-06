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
    elif model_name == 'AlexNet':
        return AlexNet()

# AlexNet for MNIST
class AlexNet(nn.Module): 
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(11, 11), stride=4, padding=2), # 224x224 -> 55x55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0), # 55x55 -> 27x27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2), # 27x27 -> 27x27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0), # 27x27 -> 13x13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1), # 13x13 -> 13x13
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),  # 13x13 -> 13x13
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),  # 13x13 -> 13x13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=0), # 13x13 -> 6x6
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
