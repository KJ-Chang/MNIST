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
            nn.Dropout2d(p=0.5),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
# VGG_E for MNIST
class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=1), # 224x224 -> 224x224
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1), # 224x224 -> 224x224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # 224x224 -> 112x112
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1), # 112x112 -> 112x112
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1), # 112x112 -> 112x112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # 112x112 -> 56x56
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1), # 56x56 -> 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1), # 56x56 -> 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1), # 56x56 -> 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1), # 56x56 -> 56x56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # 56x56 -> 28x28
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1), # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1), # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1), # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1), # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # 28x28 -> 14x14
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1), # 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1), # 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1), # 14x14 -> 14x14
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1), # 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # 14x14 -> 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        

def get_model(model_name = 'LeNet'):
    if model_name == 'LeNet':
        return LeNet()
    elif model_name == 'AlexNet':
        return AlexNet()
    elif model_name == 'VGG':
        return VGG()