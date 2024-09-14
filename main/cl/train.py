import numpy as np
import torch


def train(X, y, model, epoch, optimizer, criterion, data_len):
    
    loss_sum = 0.0
    model.train()

    for epoch in range(epoch):
        for i in range(data_len):

            X_batch = torch.tensor(X[i]).unsqueeze(0)
            y_batch = torch.tensor(y[i]).unsqueeze(0)

            optimizer.zero_grad()

            y_batch_pred = model(X_batch)
            loss = criterion(y_batch_pred, y_batch)
            
            loss.backward()
            optimizer.step()


            loss_sum += loss.item()
            if (i % 100 == 0) and (i != 0):
                print(f"Iter: {data_len*epoch+i}, Loss: {loss_sum/100}")
                loss_sum = 0

    torch.save(model.state_dict(), f'./main/cl/model.pth')
