import numpy as np
import torch
import torch.nn as nn


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5, 5), stride=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(5, 5), stride=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(5, 5), stride=1)

        self.conv4 = nn.Conv2d(16, 64, kernel_size=(5, 5), stride=2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=2)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=2)
        
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 4)

        self.relu = nn.ReLU()

        self._init_weight()

    
    def _init_weight(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.01)

    
    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool(x)

        x = x.view(x.shape[0], -1)
        x = self.fc2(self.relu(self.fc1(x)))

        return x