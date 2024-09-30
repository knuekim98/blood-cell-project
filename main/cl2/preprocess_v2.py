import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


def load_data(train_batch_size, test_batch_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.678, 0.641, 0.660), std=(0.260, 0.259, 0.257)),
        transforms.Resize((224, 224)),
    ])

    train_data = ImageFolder("./dataset/dataset2-master/images/TRAIN", transform=transform)
    test_data = ImageFolder("./dataset/dataset2-master/images/TEST", transform=transform)
    
    test_data_len = 1000
    val_subset, test_subset = random_split(test_data, [len(test_data)-test_data_len, test_data_len], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader