import h5py
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, name):
        self.X = []
        self.y = []
        with h5py.File("./main/cl2/dataset.h5", 'r') as file:
            self.X = file[f'x_{name}'][()].astype(np.float32)
            self.y = file[f'y_{name}'][()].astype(np.float32)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.y)
