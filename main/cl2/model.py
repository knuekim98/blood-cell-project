import numpy as np
import torch
import torch.nn as nn


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(16, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(16, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.5),

        )

        self.fc = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

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

        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x