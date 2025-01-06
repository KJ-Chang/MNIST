from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from config import * 


def get_dataloader():
    transfrom = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(
        root='./Dataset',
        train=True,
        download=True,
        transform=transfrom,
    )

    test_data = datasets.MNIST(
        root='./Dataset',
        train=False,
        download=True,
        transform=transfrom,
    )

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_dataloader, test_dataloader





