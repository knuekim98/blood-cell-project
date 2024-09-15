import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import CNNmodel
from read_dataset import CustomDataset


def accuracy(y_pred, y):
    y_pred = torch.argmax(y_pred, axis=1)
    y = torch.argmax(y, axis=1)
    correct = (y_pred == y).float()
    return correct.mean().item()


model = CNNmodel()
model.load_state_dict(torch.load('./main/cl2/model.pth'))
model.eval()

test_dataloader = DataLoader(CustomDataset('TEST'), batch_size=1000, shuffle=True)
cnt = 0

for i, (X, y) in enumerate(test_dataloader):
    y_pred = model(X)
    y_pred_prob = F.softmax(y_pred, dim=1)
    y_pred_label = y_pred_prob.argmax(1)

    acc = accuracy(y_pred_prob, y)
    print(acc)

    cm = confusion_matrix(y.argmax(1), y_pred_label)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

    break