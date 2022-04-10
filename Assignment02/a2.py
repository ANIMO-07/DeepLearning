# %% Imports

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy


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



# %%

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# %%

model = keras.Sequential([
	keras.layers.InputLayer(input_shape = (784, )),
	keras.layers.Dense(32, activation='relu'),
	keras.layers.Dense(15, activation='relu'),
	keras.layers.Dense(3, activation='relu'),
	keras.layers.Dense(5, activation='softmax')
])

optzr = keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
es = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, verbose=2)
model.compile(optimizer=optzr, loss='categorical_crossentropy', metrics=['accuracy'])
out = model.fit(xtrain, ytrain, validation_data=(xval, yval), batch_size=32, verbose=2, epochs=20, callbacks=[es])


# %%
