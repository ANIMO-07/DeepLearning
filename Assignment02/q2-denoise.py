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

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# %% 

l = []
def acti(x):
    a_list = [1, 0]
    distribution = [0.2, 0.8]
    random_number = random.choices(a_list, distribution)
    y = random_number*np.random.normal(size = 1)
    l.append(y[0]+x)
    return y[0]+x


#%%
input_img = keras.Input(shape=(784,))
encoded = keras.layers.Dense(64, activation='sigmoid')(input_img)
decoded = keras.layers.Dense(784, activation='linear')(encoded)

autoencoder = keras.Model(input_img, decoded)
encoder = keras.Model(input_img, encoded)

test_model = keras.Input(shape=(64,))

encoded_input = keras.Input(shape=(64,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mse')


# %%

input_img1 = keras.Input(shape=(784,))
add_noise = keras.layers.Dense(784, activation=acti,kernel_initializer='ones',bias_initializer='zeros')(input_img1)
encoder1 = keras.Model(input_img, add_noise)


encoder1.compile(optimizer='adam', loss='mse')

for i in range(1):
    xtrain1 = encoder1.predict(xtrain)
    # autoencoder.fit(xtrain1, xtrain, epochs=1, batch_size=32, validation_data=(xval, xval))


# %%

print(l)
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

input_img = keras.Input(shape=(784,))
encoded1 = keras.layers.Dense(256, activation='sigmoid')(input_img)
encoded2 = keras.layers.Dense(32, activation='sigmoid')(encoded1)
encoded3 = keras.layers.Dense(256,activation='sigmoid')(encoded2)
decoded = keras.layers.Dense(784,activation='linear')(encoded3)


autoencoder = keras.Model(input_img, decoded)
encoder = keras.Model(input_img, encoded2)

encoded_input2 = keras.Input(shape=(32,))

decoder_se_pehele = autoencoder.layers[-2]
decoder_layer = autoencoder.layers[-1]

decoder = keras.Model(encoded_input2,decoder_layer(decoder_se_pehele(encoded_input2)))


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

def acti():
    a_list = [1, 0]
    distribution = [0.2, 0.8]
    random_number = random.choices(a_list, distribution)
    print(random_number)
    y = np.random.normal(size = 1)
    return y[0]

acti()
# %%
