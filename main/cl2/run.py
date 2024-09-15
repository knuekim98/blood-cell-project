import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from read_dataset import CustomDataset
from model import CNNmodel
from train import train
from eval import evaluation


# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNmodel().to(device)
summary(model, input_size=(3, 240, 320), batch_size=1)


# hyperparameter
batch_size = 64
epoch = 10
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
#schedular = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)
criterion = nn.CrossEntropyLoss()


# train
train_dataloader = DataLoader(CustomDataset("TRAIN"), batch_size=batch_size, shuffle=True)
print("Loading Data Complete")

model_trained = train(train_dataloader, model, epoch, optimizer, criterion, device)


# eval
evaluation(model=model_trained)