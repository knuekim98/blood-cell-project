import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from preprocess_v2 import load_data
from model import CNNmodel
from train import train
from eval import evaluation


# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNmodel().to(device)
summary(model, input_size=(3, 224, 224), batch_size=1)


# hyperparameter
batch_size = 16
epochs = 50
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
schedular = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1**x)
criterion = nn.CrossEntropyLoss()


# load data
print("Load Datasets")
train_loader, val_loader, test_loader = load_data(batch_size, 100)


# train
print("Train Model")
model_name = train(train_loader, val_loader, model, epochs, optimizer, schedular, criterion, device)


# eval
print("Inference")
evaluation(test_loader, model, model_name, device)