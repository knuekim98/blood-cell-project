import h5py
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image


def make_dataset(name):

    category = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
    X = []
    y = []

    for i in tqdm(range(len(category))):
        for f in os.listdir(f"./dataset/dataset2-master/images/{name}/{category[i]}"):
            img = Image.open(f"./dataset/dataset2-master/images/{name}/{category[i]}/"+f)
            img = np.array(img)
            img = np.moveaxis(img, -1, 0)

            # normalize
            img = (img.astype(np.float16)/255)*2-1

            assert img.shape == (3, 240, 320)
            
            label = np.zeros(len(category))
            label[i] = 1
            X.append(img)
            y.append(label)        


    with h5py.File('./main/cl2/dataset.h5', 'a') as file:
        file.create_dataset(f"x_{name}", data=X, dtype=np.float16, compression="gzip")
        file.create_dataset(f"y_{name}", data=y, dtype=np.uint8, compression="gzip")

    print(len(X), len(y))


make_dataset('TRAIN')
make_dataset('TEST')
print("done")