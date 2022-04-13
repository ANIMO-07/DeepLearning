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



# %% For Hardware Acceleration

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# %%

model = keras.Sequential([
	keras.layers.InputLayer(input_shape = (784, )),
	keras.layers.Dense(78, activation='relu', name="HiddenLayer1"),
	keras.layers.Dense(39, activation='relu', name="HiddenLayer2"),
	keras.layers.Dense(20, activation='relu', name="HiddenLayer3"),
	keras.layers.Dense(5, activation='softmax', name="OuputLayer")
])

startTime = time.time()
# optzr = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
# optzr = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.99, momentum=0.9, epsilon=1E-8)
optzr = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1E-8)
es = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1E-4, verbose=2, patience=5)
tb = keras.callbacks.TensorBoard(log_dir="logs/Adam/60-20-40", histogram_freq=1)
model.compile(optimizer=optzr, loss='categorical_crossentropy', metrics=['accuracy'])
out = model.fit(xtrain, ytrain, validation_data=(xval, yval), batch_size=32, verbose=2, epochs=1000, callbacks=[es, tb])
print("Total time taken =", time.time() - startTime)



# %%


from sklearn.metrics import accuracy_score, confusion_matrix

trainpred = model.predict(xtrain)
trainpred = np.rint(trainpred)
print("Train Accuracy:", accuracy_score(ytrain, trainpred))
print("Confusion Matrix")
print(confusion_matrix(tf.argmax(ytrain, axis=1), tf.argmax(trainpred, axis=1)))

testpred = model.predict(xtest)
testpred = np.rint(testpred)
print("\nTest Accuracy:", accuracy_score(ytest, testpred))
print("Confusion Matrix")
print(confusion_matrix(tf.argmax(ytest, axis=1), tf.argmax(testpred, axis=1)))

# %%
