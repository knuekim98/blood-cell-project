import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from model import CNNmodel
from preprocess_v2 import load_data


def accuracy(y_pred, y):
    y_pred = torch.argmax(y_pred, axis=1)
    y = torch.argmax(y, axis=1)
    correct = (y_pred == y).float()
    return correct.mean().item()


def evaluation(dataloader, model, filename):
    model.load_state_dict(torch.load(filename, weights_only=True))
    model.eval()

    total_y_label = []
    total_y_pred_label = []
    mean_acc = 0.

    for i, (X, y) in tqdm(enumerate(dataloader)):
        y_pred = model(X)
        y_pred_prob = F.softmax(y_pred, dim=1)
        y_pred_label = y_pred_prob.argmax(1)

        total_y_label.append(y)
        total_y_pred_label.append(y_pred_label)

        acc = accuracy(y_pred_prob, F.one_hot(y, num_classes=4).float())
        mean_acc += acc

    print(f"Acc: {mean_acc/len(dataloader)}")

    cm = confusion_matrix(torch.cat(total_y_label), torch.cat(total_y_pred_label))
    ConfusionMatrixDisplay(cm).plot()
    plt.show()


if __name__ == '__main__':
    _, _, test_loader = load_data(16, 100)
    evaluation(test_loader, CNNmodel(), "./main/cl2/model/model-1726472613.pth")
