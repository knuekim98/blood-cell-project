import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from read_dataset import read_dataset
from model import CNNmodel
from train import train


# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNmodel().to(device)
summary(model, input_size=(3, 480, 640), batch_size=1)


# hyperparameter
epoch = 10
lr = 1e-5
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MultiLabelSoftMarginLoss()


# train
DATA_LEN = 365
X, y = read_dataset(DATA_LEN)

train(X, y, model, epoch, optimizer, criterion, DATA_LEN)
