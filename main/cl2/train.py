import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

from eval import accuracy


def model_pass(model, X, y, criterion, device, backprop=False, optimizer=None):
    X = X.to(device)
    y = F.one_hot(y, num_classes=4).float().to(device)

    y_pred = model(X)
    loss = criterion(y_pred, y)

    if backprop:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = accuracy(F.softmax(y_pred, dim=1), y)
    return loss.item(), acc


def train(train_loader, val_loader, model, epochs, optimizer, schedular, criterion, device):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        train_loss_mean = 0.
        train_acc_mean = 0.
        val_loss_mean = 0.
        val_acc_mean = 0.

        model.train()
        for _, (X, y) in tqdm(enumerate(train_loader)):
            loss, acc = model_pass(model, X, y, criterion, device, backprop=True, optimizer=optimizer)
            train_loss_mean += loss
            train_acc_mean += acc

        model.eval()
        for _, (X, y) in enumerate(val_loader):
            with torch.no_grad():
                loss, acc = model_pass(model, X, y, criterion, device)
                val_loss_mean += loss
                val_acc_mean += acc
                

        schedular.step()

        train_loss_mean /= len(train_loader)
        train_acc_mean /= len(train_loader)
        val_loss_mean /= len(val_loader)
        val_acc_mean /= len(val_loader)
        train_losses.append(train_loss_mean)
        val_losses.append(val_loss_mean)
        print(f"epoch {epoch+1}")
        print(f"train_loss {train_loss_mean:.6f}, train_acc {train_acc_mean:.6f}")
        print(f"val_Loss {val_loss_mean:.6f}, val_acc {val_acc_mean:.6f}")


    model_name = f'./main/cl2/model/model-{int(time())}.pth'
    torch.save(model.state_dict(), model_name)
    print(f"Saved as {model_name}")

    plot_x = [i+1 for i in range(epochs)]
    plt.plot(plot_x, train_losses, label="train_losses")
    plt.plot(plot_x, val_losses, label="val_losees")
    plt.ylim(0, max(train_losses+val_losses)*1.2)
    plt.legend()
    plt.show()

    return model_name
