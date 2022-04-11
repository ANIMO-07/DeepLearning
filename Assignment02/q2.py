# %% Imports

import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow import keras



# %% Importing Data

def getdata(folder):
	numbers = [1, 5, 6, 7, 9]
	data = []

	for k in range(5):
		for i in os.listdir(f"./Group_16/{folder}/{numbers[k]}"):
			if i.endswith(".jpg"):
				image = Image.open(f"./Group_16/{folder}/{numbers[k]}/{i}")
				img = np.asarray(image).flatten()
				label = [0, 0, 0, 0, 0]
				label[k] = 1
				img = np.append(img, label)
				data.append(img)
	
	random.shuffle(data)
	data = pd.DataFrame(data)

	return data



# %%

train = getdata("train")
val = getdata("val")
test = getdata("test")


# %%

xtrain, ytrain = train.iloc[: , :-5], train.iloc[: , -5:]
xval, yval = val.iloc[: , :-5], val.iloc[: , -5:]
xtest, ytest = test.iloc[: , :-5], test.iloc[: , -5:]

xtrain = xtrain/255
xval = xval/255
xtest = xtest/255


# %% For Hardware Acceleration

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# %% 

input_img = keras.Input(shape=(784,))
encoded = keras.layers.Dense(32, activation='sigmoid')(input_img)
decoded = keras.layers.Dense(784, activation='linear')(encoded)

autoencoder = keras.Model(input_img, decoded)
encoder = keras.Model(input_img, encoded)

encoded_input = keras.Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mse')


# %%

autoencoder.fit(xtrain, xtrain, epochs=50, batch_size=32, validation_data=(xval, xval))


# %%

encoded_imgs = encoder.predict(xtest)
decoded_imgs = decoder.predict(encoded_imgs)

# %%

import matplotlib.pyplot as plt

n = 20  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(xtest.iloc[i].values.reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# %%
