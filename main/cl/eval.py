import torch
import matplotlib.pyplot as plt
from read_dataset import read_dataset
from model import CNNmodel


model = CNNmodel()
model.load_state_dict(torch.load('./main/cl/model.pth'))
model.eval()

X_test, y_test = read_dataset(15)

print(y_test)
print(model(torch.tensor(X_test)))
