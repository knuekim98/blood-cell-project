import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from read_dataset import CustomDataset
from model import CNNmodel
from train import train


# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNmodel().to(device)
summary(model, input_size=(3, 240, 320), batch_size=1)


# hyperparameter
batch_size = 100
epoch = 10
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# train
train_dataloader = DataLoader(CustomDataset("TRAIN"), batch_size=batch_size, shuffle=True)
print("Loading Data Complete")

train(train_dataloader, model, epoch, optimizer, criterion)
