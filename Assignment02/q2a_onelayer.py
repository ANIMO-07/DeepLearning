# %% Imports

import os
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


# %% Single Layer Autoencoder

numnodes = 32

input_img = keras.Input(shape=(784,))
encoded = keras.layers.Dense(numnodes, activation='sigmoid')(input_img)
decoded = keras.layers.Dense(784, activation='linear')(encoded)

autoencoder = keras.Model(input_img, decoded)
encoder = keras.Model(input_img, encoded)

encoded_input = keras.Input(shape=(numnodes,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

es = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1E-4, verbose=2, patience=1)
tb = keras.callbacks.TensorBoard(log_dir="logs/autoencoder/one_layer", histogram_freq=1)

autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# %%

autoencoder.fit(xtrain, xtrain, epochs=1000, batch_size=32, validation_data=(xval, xval), callbacks = [es, tb])


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

from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

encoded_test = encoder.predict(xtest)
decoded_test = decoder.predict(encoded_test)

print("Test Reconstruction Error for chosen best model:")
print(mean_squared_error(xtest, decoded_test))


# %%

trainpred = model.predict(encoded_train)
trainpred = np.rint(trainpred)
print("Train Accuracy:", accuracy_score(ytrain, trainpred))
print("Confusion Matrix")
print(confusion_matrix(tf.argmax(ytrain, axis=1), tf.argmax(trainpred, axis=1)))

valpred = model.predict(encoded_val)
valpred = np.rint(valpred)
print("Val Accuracy:", accuracy_score(yval, valpred))
print("Confusion Matrix")
print(confusion_matrix(tf.argmax(yval, axis=1), tf.argmax(valpred, axis=1)))

testpred = model.predict(encoded_test)
testpred = np.rint(testpred)
print("\nTest Accuracy:", accuracy_score(ytest, testpred))
print("Confusion Matrix")
print(confusion_matrix(tf.argmax(ytest, axis=1), tf.argmax(testpred, axis=1)))


# %%
