import numpy as np
import torch
import torch.nn as nn


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # -> 112 

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # -> 56
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # -> 28  
            nn.Dropout(0.2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # -> 14
            nn.Dropout(0.2),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), # -> 7
            nn.Dropout(0.5),
            
        )

        self.fc = nn.Sequential(

            nn.Linear(32*7*7, 256),
            nn.ReLU(),

            nn.Linear(256, 32),
            nn.ReLU(),

            nn.Linear(32, 4),
        )

        self._init_weight()

    
    def _init_weight(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x