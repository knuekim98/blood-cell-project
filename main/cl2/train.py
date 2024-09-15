import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval import accuracy


def train(dataloader, model, epoch, optimizer, criterion):
    
    loss_values = []
    acc_values = []
    model.train()

    for epoch in range(epoch):
        for i, data in tqdm(enumerate(dataloader)):
            
            X, y = data
            optimizer.zero_grad()

            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            loss.backward()
            optimizer.step()

            acc = accuracy(F.softmax(y_pred, dim=1), y)

        loss_values.append(loss.detach().clone().numpy().item())
        acc_values.append(acc)

        print(f"epoch {epoch+1}: Loss {loss_values[-1]:.6f}, Acc {acc:.6f}")

    torch.save(model.state_dict(), f'./main/cl2/model.pth')
