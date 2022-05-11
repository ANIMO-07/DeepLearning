# %% Imports

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# %% Importing Data

img = Image.open(f"./Group_16/train/butterfly/image_0001.jpg")
img = img.resize((224,224))
img = img.convert('L')
butterfly = np.asarray(img)

img = Image.open(f"./Group_16/train/kangaroo/image_0002.jpg")
img = img.resize((224,224))
img = img.convert('L')
kangaroo = np.asarray(img)

img = Image.open(f"./Group_16/train/Leopards/image_0004.jpg")
img = img.resize((224,224))
img = img.convert('L')
leopard = np.asarray(img)


# %% Initialising filter using Kaiming Initialisation

np.random.seed(42)
filter = [[np.random.normal(0, 2/224) for i in range(3)] for j in range(3)]
print(filter)


# %%

output = []

for i in range(0, 222):
    output.append([])
    for j in range(0, 222):
        submatrix = butterfly[i:i+3, j:j+3]
        product = np.multiply(submatrix, filter)
        sum = np.sum(product)
        output[i].append(sum)

output = np.array(output)

# %%

plt.imshow(output, cmap="gray")


# %%

plt.imshow(butterfly, cmap="gray")


# %%
