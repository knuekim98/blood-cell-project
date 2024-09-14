import h5py
import numpy as np

def read_dataset(data_len):
    X = []
    y = []

    with h5py.File("./main/cl/dataset.h5", 'r') as file:
        X = file['x'][:data_len]
        y = file['y'][:data_len]

        return X, y