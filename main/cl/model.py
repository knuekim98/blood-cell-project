import numpy as np
import torch
import torch.nn as nn


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()

        self.conv1 = nn.Conv2d(3, 40, kernel_size=(10, 10), stride=2)
        self.conv2 = nn.Conv2d(40, 20, kernel_size=(10, 10), stride=2)
        self.conv3 = nn.Conv2d(20, 10, kernel_size=(5, 5), stride=2)
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = nn.Linear(480, 100)
        self.fc2 = nn.Linear(100, 5)

        self.relu = nn.ReLU()

    
    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.shape[0], -1)
        x = self.fc2(self.relu(self.fc1(x)))

        return x