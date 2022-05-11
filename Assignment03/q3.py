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

def getdata(folder):
    classes = ["butterfly", "kangaroo", "Leopards"]
    data = []

    for k in range(3):
        for i in os.listdir(f"./Group_16/{folder}/{classes[k]}"):
            if i.endswith(".jpg"):
                image = Image.open(f"./Group_16/{folder}/{classes[k]}/{i}")
                img = image.resize((224,224))
                img = np.asarray(img)
                label = np.array([0, 0, 0])
                label[k] = 1
                img = np.array([img, label])
                data.append(img)
	
    random.shuffle(data)
    return np.array(data)

# %%

train = getdata("train")
val = getdata("val")
test = getdata("test")

# %%

def extract(data):
    xtrain = []
    ytrain = []
    for i in range(len(data)):
        if(len(data[i][0].shape)==3):
            xtrain.append(data[i][0])
            ytrain.append(data[i][1])

    return np.array(xtrain), np.array(ytrain)

# %%

xtrain, ytrain = extract(train)
xval, yval = extract(val)
xtest, ytest = extract(test)

# %% Importing Data

img = Image.open(f"./Group_16/train/butterfly/image_0001.jpg")
img = img.resize((224, 224))
butterfly = np.asarray(img)

img = Image.open(f"./Group_16/train/kangaroo/image_0002.jpg")
img = img.resize((224, 224))
kangaroo = np.asarray(img)

img = Image.open(f"./Group_16/train/Leopards/image_0004.jpg")
img = img.resize((224, 224))
leopard = np.asarray(img)

# %%

class convolutional_layers:
    def init(self):
        self.filters = []

    def filter(self, height, width, depth, nl, no_of_filters):
        if(depth == 0):
            return np.array([[[np.random.normal(0, 2/nl) for i in range(height)] for j in range(width)] for k in range(no_of_filters)])
        else:
            return np.array([[[[np.random.normal(0, 2/nl) for i in range(height)] for j in range(width)] for d in range(depth)]for k in range(no_of_filters)])

    def relu(self, x):
        if(x > 0):
            return x
        else:
            return 0

    def convolve(self, input, filter):
        output = []

        if(len(input.shape) == 2):
            for i in range(0, input.shape[0]-filter.shape[0]+1):
                output.append([])
                for j in range(0, input.shape[1]-filter.shape[1]+1):
                    submatrix = input[i:i+3, j:j+3]
                    product = np.multiply(submatrix, filter)
                    sum = np.sum(product)
                    output[i].append(self.relu(sum))
        else:
            for i in range(0, input.shape[0]-filter.shape[1]+1):
                output.append([])
                for j in range(0, input.shape[1]-filter.shape[2]+1):
                    submatrix = input[i:i+3, j:j+3, :]
                    product = np.multiply(filter, submatrix)
                    sum = np.sum(product)
                    output[i].append(self.relu(sum))

        output = np.array(output)
        return output

    def convolve2(self, input, filter):
        output = []

        if(len(input.shape) == 2):
            for i in range(0, input.shape[0]-filter.shape[0]+1):
                output.append([])
                for j in range(0, input.shape[1]-filter.shape[1]+1):
                    submatrix = input[i:i+3, j:j+3]
                    product = np.multiply(submatrix, filter)
                    sum = np.sum(product)
                    output[i].append(self.relu(sum))
        else:
            for i in range(0, input.shape[1]-filter.shape[1]+1):
                output.append([])
                for j in range(0, input.shape[2]-filter.shape[2]+1):
                    submatrix = input[:,i:i+3, j:j+3]
                    product = np.multiply(submatrix, filter)
                    sum = np.sum(product)
                    output[i].append(self.relu(sum))
        output = np.array(output)
        return output

    def layer(self, input, tag):
        output = []
        filters = self.filters
        for i in filters:
            if(tag == 'input'):
                output.append(self.convolve(input, i))
            else:
                output.append(self.convolve2(input, i))
        
        output = np.array(output)
        return output

    def initialize_filter(self, height, width, depth, nl, no_of_filters):
        self.filters = self.filter(height, width, depth, nl, no_of_filters)


# %%

def pooling(dim, feature_map, stride):
    output = []
    sz = feature_map.shape
    new_W = int(((sz[1]-dim)/stride)+1)
    new_H = int(((sz[2]-dim)/stride)+1)
    new_D = sz[0]

    for d in range(new_D):
        layer_at_d = []
        for i in range(new_W):
            row_at_i = []
            for j in range(new_H):
                submatrix = feature_map[d, i:i+dim, j:j+dim]
                maxi = np.max(submatrix)
                row_at_i.append(maxi)
            layer_at_d.append(row_at_i)
        output.append(layer_at_d)
    output = np.array(output)
    return output


# %%

class fully_connected_layers:
    def init(self):
        self.weights = []
        self.bias = []

    def relu(self, arr):
        tmp = []
        for an in arr:
            if(an <= 0):
                tmp.append(0)
            else:
                tmp.append(an)
        return tmp

    def softMax(self, arr):
        exps = np.exp(arr)
        exps /= np.sum(arr)
        return exps

    def dense_forward_pass(self, input):
        bias = self.bias
        wts = self.weights
        output_nodes = wts[0].size
        n = len(input)
        output = []
        for j in range(output_nodes):
                output.append(0)
        for i in range(n):
            for j in range(output_nodes):
                print(j, output_nodes)
                output[j] += (wts[i][j]*input[i]) + bias[j] 
        output = self.relu(output)    
        output = np.array(output)
        return output
    
    def output_layer(self, input):
        output = self.softMax(input)
        
        return np.array(output)

    def initialize_weights(self, input_nodes, output_nodes, n):
        self.weights = np.array([[np.random.normal(0, 1/math.pow(n, 0.5)) for i in range(output_nodes)] for j in range(input_nodes)])
        self.bias = np.array([1 for i in range(output_nodes)])

# %%

class CNN:

    def fit(self, X, Y, epochs=30, eta = 0.4, thresh = 1E-3):
		
        layer1 = convolutional_layers()
        layer1.initialize_filter(3, 3, 3, 224*224, 32)

        layer2 = convolutional_layers()
        layer2.initialize_filter(3, 3, 32, 222*222, 64)

        layer3 = fully_connected_layers()
        layer3.initialize_weights(3069504, 3, 100)

        layer4 = fully_connected_layers()
        
        output = []

        for i in X:
            output = layer1.layer(i, 'input')
            output2 = layer2.layer(output, 'not')
            pool_output = pooling(2, output2, 1)
            next_layer = pool_output.flatten()
            output3 = layer3.dense_forward_pass(next_layer)
            output4 = layer4.output_layer(output3)
            output.append(output4)
            print(output4)

        return output
    
#%%

print(butterfly[0:3,0:3,:].shape)

# %%

model = CNN()

print(model.fit(xtrain[:5], ytrain[:5]))
# %%
