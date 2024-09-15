import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from time import time

from eval import accuracy


def train(dataloader, model, epoch, optimizer, criterion, device):
    
    model.train()

    for epoch in range(epoch):

        loss_mean = 0.0
        acc_mean = 0.0

        for i, (X, y) in tqdm(enumerate(dataloader)):

            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            loss.backward()
            optimizer.step()

            acc = accuracy(F.softmax(y_pred, dim=1), y)

            loss_mean += loss.item()
            acc_mean += acc

        loss_mean /= len(dataloader)
        acc_mean /= len(dataloader)
        print(f"epoch {epoch+1}: Loss {loss_mean:.6f}, Acc {acc_mean:.6f}")


    model_name = f'./main/cl2/model/model-{int(time())}.pth'
    torch.save(model.state_dict(), model_name)
    print(f"Saved as {model_name}")

    return model
