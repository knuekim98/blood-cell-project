import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def create_datasets(train_batch_size, test_batch_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    train_data = ImageFolder("./dataset/dataset2-master/images/TRAIN", transform=transform)
    test_data = ImageFolder("./dataset/dataset2-master/images/TEST", transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader
