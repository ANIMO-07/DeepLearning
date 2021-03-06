# %% Imports

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

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

class Noise(keras.layers.Layer):
	def call(self, inputs, training=True):
		if random.random() <= 0.4:
			# print(tf.shape(inputs))
			# return inputs + self._customNoise(0.2)
			# for i in range(inputs.shape):
			noisy = inputs + tf.random.normal(tf.shape(inputs), mean=0, stddev=1)
			# return inputs + tf.random.normal([11385,784], mean=0, stddev=0.5)
			return tf.keras.backend.in_train_phase(noisy, inputs, training=training)
		else:
			return inputs


# %%

class Noise(keras.layers.Layer):
	def _customNoise(self, percent=0.2):
		noise = [0]*784
		for i in range(784):
			if random.random() <= percent:
				noise[i] += np.random.normal(0, 0.5)
		return noise

	def call(self, inputs, training=False):
		if training:
			return inputs + self._customNoise(0.2)
		else:
			return inputs


# %% Single Layer Autoencoder

numnodes = 32

input_img = keras.Input(shape=(784,))
noisy = Noise()(input_img)
encoded = keras.layers.Dense(numnodes, activation='sigmoid')(noisy)
decoded = keras.layers.Dense(784, activation='linear')(encoded)

autoencoder = keras.Model(input_img, decoded)
encoder = keras.Model(input_img, encoded)

encoded_input = keras.Input(shape=(numnodes,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

es = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1E-4, verbose=2, patience=1)
# tb = keras.callbacks.TensorBoard(log_dir="logs/autoencoder/one_layer", histogram_freq=1)

autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# %%

autoencoder.fit(xtrain, xtrain, epochs=1000, batch_size=32, validation_data=(xval, xval), callbacks = [es])

# %%

from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

encoded_test = encoder.predict(xtest)
decoded_test = decoder.predict(encoded_test)

print("Test Reconstruction Error:")
print(mean_squared_error(xtest, decoded_test))


# %%

encoded_train = encoder.predict(xtrain)
decoded_train = decoder.predict(encoded_train)

encoded_val = encoder.predict(xval)
decoded_val = decoder.predict(encoded_val)

# %%

import matplotlib.pyplot as plt

n = 20

plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(xtrain.iloc[i].values.reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# %% Running the classifier on the encoded representation

model = keras.Sequential([
	keras.layers.InputLayer(input_shape = (numnodes, )),
	keras.layers.Dense(78, activation='relu', name="HiddenLayer1"),
	keras.layers.Dense(39, activation='relu', name="HiddenLayer2"),
	keras.layers.Dense(20, activation='relu', name="HiddenLayer3"),
	keras.layers.Dense(5, activation='softmax', name="OutputLayer")
])

optzr = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1E-8)
model.compile(optimizer=optzr, loss='categorical_crossentropy', metrics=['accuracy'])
out = model.fit(encoded_train, ytrain, validation_data=(encoded_val, yval), batch_size=32, verbose=2, epochs=1000, callbacks=[es])

# %%

weights = encoder.layers[-1].weights
arr = np.array(weights[0]).T

for j in range(numnodes):
    plt.imshow(arr[j].reshape(28,28))
    plt.title(f"Hidden Neuron: {j}")
    plt.show()


# %%
