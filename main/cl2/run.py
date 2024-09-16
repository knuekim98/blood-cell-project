import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from preprocess_v2 import create_datasets
from model import CNNmodel
from train import train
from eval import evaluation


# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNmodel().to(device)
summary(model, input_size=(3, 240, 320), batch_size=1)


# hyperparameter
batch_size = 16
epoch = 15
lr = 0.0005 #0.00005
optimizer = optim.Adam(model.parameters(), lr=lr)
#schedular = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95**epoch)
criterion = nn.CrossEntropyLoss()


# load data
print("Load Datasets")
train_loader, test_loader = create_datasets(batch_size, 100)


# train
print("Train Model")
model_name = train(train_loader, model, epoch, optimizer, criterion, device)


# eval
print("Inference")
evaluation(test_loader, model, model_name)