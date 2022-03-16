# %% Imports

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join
from itertools import combinations
import math
from sklearn.metrics import confusion_matrix, accuracy_score



# %% Colecting LS Data

LSpath = "./data/Group16/Classification/LS_Group16/"
classes = ["Class1.txt", "Class2.txt", "Class3.txt"]

dfs = []

for i, c in enumerate(classes):
	df = pd.read_csv(join(LSpath, c), names = ["x", "y"], sep = " ", dtype = {"x" : "float", "y" : "float"}, engine = "python")
	df["class"] = i 
	dfs.append(df)

ls_data = pd.concat(dfs, axis=0).reset_index(drop=True) 




# %% Collecting NLS Data

nls_data = pd.read_csv("./data/Group16/Classification/NLS_Group16.txt", names = ["x", "y"], sep = " ", 
					dtype = {"x" : "float", "y" : "float"}, engine = "python", skiprows = 1, index_col = False)

nls_data["class"] = [0]*500 + [1]*500 + [2]*500




# %%

class perceptron:
	def __init__(self):
		self.weightvec = []
		self.models = []
		self.errorvsepochs = []
	
	def _sigmoid(self, an):
		return 1/(1+math.pow(math.exp(1),-an))
 
	def _give_error(self, yn, sn):
		En = math.pow((yn-sn),2)/2
		return En
	
	def _differentiate(self, an):
		diff = self._sigmoid(an) * (1 - self._sigmoid(an))
		return diff
	
	def _get_weights(self, trainx, trainy, wts, eta, epochs):
		N = len(trainx)

		errvsepochs = []
		for i in range(epochs):
			total_epoch_error = 0

			for j in range(N):
				activation_value = sum(np.multiply(wts, trainx[j]))

				predicted_output = self._sigmoid(activation_value)
				true_output = trainy[j]

				instantaneous_error = self._give_error(true_output, predicted_output)
				total_epoch_error += instantaneous_error

				delta_w = eta * (true_output-predicted_output) * self._differentiate(activation_value) * trainx[j]
	
				# print(i, j, predicted_output, true_output, instantaneous_error, delta_w)
				wts = np.add(wts, delta_w)
			
			errvsepochs.append(total_epoch_error/N)
			# print(wts)
			# print("For Epoch:", i, "     Calculated Average Error =", total_epoch_error/N)

		self.errorvsepochs.append(errvsepochs)
		return wts

	def fit(self, X, y, initial_vec_val = 0,  epochs=30, eta = 0.4):
		classes = pd.unique(y)
		models = list(combinations(classes, 2))

		dimensions = len(X.columns)
		t = len(models)						# Total number of perceptrons required for multiclass classification

		weightvec = [[initial_vec_val] * (dimensions + 1) for i in range(t)]

		for m in range(t):
			inputx = []
			inputy = []
			
			x = X.to_numpy()

			for i in range(len(x)):
				if y[i] == models[m][0]:
					inputy.append(0)
					inputx.append(x[i])
				if y[i] == models[m][1]:
					inputy.append(1)
					inputx.append(x[i])
				
			
			# return inputx, inputy
			inputx_ = np.c_[np.ones(len(inputx)), inputx]
			weightvec[m] = self._get_weights(inputx_, inputy, weightvec[m], eta, epochs)

		self.weightvec = weightvec
		self.models = models
		print("Model Fitting Complete")
		
	def getavgerr(self):
		return self.errorvsepochs

	def getweights(self):
		return self.weightvec

	def predict(self, Xvals):
		predictions = []

		xvals_ = np.c_[np.ones(len(Xvals)), Xvals]
		weights = self.weightvec
		models = self.models
		t = len(models)

		for x in xvals_:
			outs = []
			for i in range(t):
				activation_value = sum(np.multiply(weights[i], x))
				predicted_output = self._sigmoid(activation_value)

				if predicted_output <= 0.5:
					outs.append(models[i][0])
				else:
					outs.append(models[i][1])
			
			predictions.append(max(outs, key = outs.count))
		
		return predictions


# %%

classifier = perceptron()

classifier.fit(ls_data.drop(["class"], axis = 1), ls_data["class"], epochs = 10, eta = 0.4)
classifier.getweights()
classifier.getavgerr()


# %%

output = classifier.predict(ls_data.drop(["class"], axis = 1))


# %%

print(confusion_matrix(ls_data["class"], output))
print(accuracy_score(ls_data["class"], output))



# %%
clr = ["r", "g", "b"]
plt.scatter(ls_data["x"], ls_data["y"], color = [clr[k] for k in ls_data["class"]])
plt.show()



# %%
plt.scatter(x[:, 0], x[:, 1], color = [clr[k] for k in y])
plt.show()



# %%
classifier = perceptron()

classifier.fit(nls_data.drop(["class"], axis = 1), nls_data["class"], epochs = 30, eta = 0.4)
classifier.getweights()
classifier.getavgerr()


# %%

output = classifier.predict(nls_data.drop(["class"], axis = 1))


# %%

print(confusion_matrix(nls_data["class"], output))
print(accuracy_score(nls_data["class"], output))



# %%
clr = ["r", "g", "b"]
plt.scatter(nls_data["x"], nls_data["y"], color = [clr[k] for k in nls_data["class"]])
plt.show()


# %%
