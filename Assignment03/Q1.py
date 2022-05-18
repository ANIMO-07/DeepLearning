# %% Imports

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras

# %% Importing Data

img = Image.open(f"./Group_16/train/butterfly/image_0001.jpg")
# plt.imshow(img)

img = img.resize((224,224))
# plt.imshow(butterfly)
img = img.convert('L')
butterfly = np.asarray(img)
# plt.imshow(butterfly)
# img.show()

img = Image.open(f"./Group_16/train/kangaroo/image_0002.jpg")

# plt.imshow(img)

img = img.resize((224,224))
img = img.convert('L')
kangaroo = np.asarray(img)

img = Image.open(f"./Group_16/train/Leopards/image_0004.jpg")

plt.imshow(img)

img = img.resize((224,224))
img = img.convert('L')
leopard = np.asarray(img)

filter = [[np.random.normal(0, 2/224) for i in range(3)] for j in range(3)]
# %%

filter = [[np.random.normal(0, 2/224) for i in range(3)] for j in range(3)]

print(filter)

# %%

output = []

for i in range(0, 222):
    output.append([])
    for j in range(0, 222):
        submatrix = leopard[i:i+3, j:j+3]
        product = np.multiply(submatrix, filter)
        sum = np.sum(product)
        output[i].append(sum)

output = np.array(output)
print(len(output))


# %%

print(output.shape)
plt.imshow(output)


# %%