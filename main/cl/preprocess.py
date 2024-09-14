import h5py
import numpy as np
import pandas as pd
import os
from PIL import Image


labels = pd.read_csv("./dataset/dataset-master/labels.csv")
X = []
y = []

for f in os.listdir("./dataset/dataset-master/JPEGImages/"):
    img = Image.open("./dataset/dataset-master/JPEGImages/"+f)
    img = np.array(img)
    img = np.moveaxis(img, -1, 0)

    # normalize
    img = (img.astype(np.float32)/255)*2-1

    assert img.shape == (3, 480, 640)
    
    # one-hot encoding
    img_number = int(f[11:16])
    cells = labels.loc[labels["Image"]==img_number, "Category"].item()
    if not isinstance(cells, str): continue

    #print(cells, img_number, type(cells))

    cells_one_hot = np.array([
        int('NEUTROPHIL' in cells),
        int('EOSINOPHIL' in cells),
        int('LYMPHOCYTE' in cells),
        int('MONOCYTE' in cells),
        int('BASOPHIL' in cells)
    ])

    X.append(img)
    y.append(cells_one_hot)


print(len(X), len(y))

with h5py.File('./main/cl/dataset.h5', 'w') as file:
    file.create_dataset("x", data=X)
    file.create_dataset("y", data=y)

print("done")