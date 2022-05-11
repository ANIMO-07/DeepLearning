# %% Imports

import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
import math
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# %% Importing Data

img = Image.open(f"./Group_16/train/butterfly/image_0001.jpg")
img = img.resize((224,224))
img = img.convert('L')
butterfly = np.asarray(img)
# plt.imshow(butterfly)
# img.show()

img = Image.open(f"./Group_16/train/kangaroo/image_0002.jpg")
img = img.resize((224,224))
img = img.convert('L')
kangaroo = np.asarray(img)

img = Image.open(f"./Group_16/train/Leopards/image_0004.jpg")
img = img.resize((224,224))
img = img.convert('L')
leopard = np.asarray(img)

# %%

class convolutional_layers:
	def __init__(self):
		self.filters = []
	
	def filter(self, height, width, depth, nl, no_of_filters):
		if(depth==0):
			return np.array([[[np.random.normal(0, 2/nl) for i in range(height)] for j in range(width)] for k in range(no_of_filters)])
		else:
			return np.array([[[[np.random.normal(0, 2/nl) for i in range(height)] for j in range(width)] for d in range(depth+1)]for k in range(no_of_filters)])

	def relu(self, x):
		if(x>0):
			return x
		else:
			return 0

	def convolve(self, input, filter):
		output = []
		if(len(input.shape) == 2):
			for i in range(0, input.shape[0]-filter.shape[0]):
				output.append([])
				for j in range(0, input.shape[1]-filter.shape[1]):
					submatrix = input[i:i+3, j:j+3]
					product = np.multiply(submatrix, filter)
					sum = np.sum(product)
					output[i].append(self.relu(sum))
		else:
			for i in range(0, input.shape[1]-filter.shape[1]):
				output.append([])
				for j in range(0, input.shape[2]-filter.shape[2]):
					submatrix = input[:,i:i+3, j:j+3]
					product = np.multiply(submatrix, filter)
					sum = np.sum(product)
					output[i].append(self.relu(sum))

		output = np.array(output)
		return output
    	
	def layer(self, input):
		output = []
		filters = self.filters
		for i in filters:
			output.append(self.convolve(input, i))
    	
		output = np.array(output)
		return output
	
	def initialize_filter(self, height, width, depth, nl, no_of_filters):
		self.filters = self.filter(height, width, depth, nl, no_of_filters)

# %%

layer1 = convolutional_layers()

layer1.initialize_filter(3, 3, 0, 224*224, 32)

output = layer1.layer(butterfly)

print(output)

# %%

plt.imshow(output[8])

# %%

layer2 = convolutional_layers()

layer2.initialize_filter(3, 3, 31, 224*224, 64)

output2 = layer2.layer(output)

print(output2)


# %%

plt.imshow(output2[8])


# %%

