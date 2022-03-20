# %% Imports

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join
from itertools import combinations
import math
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split



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




# %% Classification Tasks - Perceptron

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
	
	def _get_weights(self, trainx, trainy, wts, eta, epochs, thresh):
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

			if i > 0:
				if errvsepochs[i-1] - errvsepochs[i] < thresh:
					break
			# print(wts)
			# print("For Epoch:", i, "     Calculated Average Error =", total_epoch_error/N)

		self.errorvsepochs.append(errvsepochs)
		return wts

	def fit(self, X, y, initial_vec_val = 0,  epochs=30, eta = 0.4, thresh = 1E-3):
		classes = pd.unique(y)
		models = list(combinations(classes, 2))

		dimensions = len(X.columns)
		t = len(models)						# Total number of perceptrons required for multiclass classification

		weightvec = [[initial_vec_val] * (dimensions + 1) for i in range(t)]

		x = X.to_numpy()
		y = y.to_numpy()

		for m in range(t):
			inputx = []
			inputy = []

			for i in range(len(x)):
				if y[i] == models[m][0]:
					inputy.append(0)
					inputx.append(x[i])
				if y[i] == models[m][1]:
					inputy.append(1)
					inputx.append(x[i])
				
			
			# return inputx, inputy
			inputx_ = np.c_[np.ones(len(inputx)), inputx]
			weightvec[m] = self._get_weights(inputx_, inputy, weightvec[m], eta, epochs, thresh)

		self.weightvec = weightvec
		self.models = models
		print("Model Fitting Complete")
		
	def params(self):
		return self.errorvsepochs, self.weightvec, self.models

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
		
		return np.array(predictions)

	def prepareModel(self, weightvecin, modelsin):
		self.weightvec = weightvecin
		self.models = modelsin


# %% Q0

#For Linearly Separable Data

xtrainLS, xtestLS, ytrainLS, ytestLS = train_test_split(ls_data.drop(["class"], axis = 1), 
										ls_data["class"], test_size = 0.4, random_state = 1)
			# 60-40 split since validation was supposed to be considered as more test data
LSclassifier = perceptron()
LSclassifier.fit(xtrainLS, ytrainLS, epochs = 10, eta = 0.4)


#For Non-Linearly Separable Data

xtrainNLS, xtestNLS, ytrainNLS, ytestNLS = train_test_split(nls_data.drop(["class"], axis = 1), 
										nls_data["class"], test_size = 0.4, random_state = 1)
			# 60-40 split since validation was supposed to be considered as more test data
NLSclassifier = perceptron()
NLSclassifier.fit(xtrainNLS, ytrainNLS, epochs = 10, eta = 0.4)




# %% Q1

#For Linearly Separable Data

errorvsepochsLS, weightvecLS, modelsLS = LSclassifier.params()

for i in range(len(modelsLS)):
	label = "Model " + str(modelsLS[i][0]) + "/" + str(modelsLS[i][1])
	plt.plot(range(1, len(errorvsepochsLS[i]) + 1), errorvsepochsLS[i], label=label)
plt.xlabel("Epochs")
plt.ylabel("Average Error")
plt.legend(loc="upper right")
plt.title("Perceptrons for Linearly Separable Data")
plt.show()

#For Linearly Separable Data

errorvsepochsNLS, weightvecNLS, modelsNLS = NLSclassifier.params()

for i in range(len(modelsNLS)):
	label = "Model " + str(modelsNLS[i][0]) + "/" + str(modelsNLS[i][1])
	plt.plot(range(1, len(errorvsepochsNLS[i]) + 1), errorvsepochsNLS[i], label=label)
plt.xlabel("Epochs")
plt.ylabel("Average Error")
plt.legend(loc="upper right")
plt.title("Perceptrons for Non-Linearly Separable Data")
plt.show()




# %% Q2 Decision Region Plot superimposed by training data

def decision_boundary(model, X, y, density = 0.1, padding = 1):

	X = X.to_numpy()
	y = y.to_numpy()

	min1, max1 = X[:, 0].min() - padding, X[:, 0].max() + padding
	min2, max2 = X[:, 1].min() - padding, X[:, 1].max() + padding

	x1grid = np.arange(min1, max1, density)
	x2grid = np.arange(min2, max2, density)

	xx, yy = np.meshgrid(x1grid, x2grid)

	r1, r2 = xx.flatten(), yy.flatten()
	r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

	grid = np.hstack((r1,r2))

	yhat = model.predict(grid)

	zz = yhat.reshape(xx.shape)

	plt.contourf(xx, yy, zz, cmap='Paired')

	plt.scatter(X[:,0], X[:,1], c = y, s = 5)
	plt.show()


# %%

def getxy(classes, X, y):
	x = X.to_numpy()
	y = y.to_numpy()

	tempx = []
	tempy = []

	for i in range(len(x)):
		if y[i] in classes:
			tempy.append(y[i])
			tempx.append(x[i])
	
	return pd.DataFrame(tempx), pd.DataFrame(tempy)



# %% For Linearly Separable Data

for i in range(len(modelsLS)):
	title = "Decision Boundary in LSData for Model " + str(modelsLS[i][0]) + "/" + str(modelsLS[i][1])
	plt.title(title)

	tempx, tempy = getxy(modelsLS[i], xtrainLS, ytrainLS)

	tempclassifier = perceptron()
	tempclassifier.prepareModel([weightvecLS[i]], [modelsLS[i]])

	decision_boundary(tempclassifier, tempx, tempy)



# %% For Non-Linearly Separable Data

for i in range(len(modelsNLS)):
	title = "Decision Boundary in NLSData for Model " + str(modelsNLS[i][0]) + "/" + str(modelsNLS[i][1])
	plt.title(title)

	tempx, tempy = getxy(modelsNLS[i], xtrainNLS, ytrainNLS)

	tempclassifier = perceptron()
	tempclassifier.prepareModel([weightvecNLS[i]], [modelsNLS[i]])

	decision_boundary(tempclassifier, tempx, tempy, density = 0.025)

# %% Combined

plt.title("Combined Decision Boundary for Linearly Separable Data")
decision_boundary(LSclassifier, xtrainLS, ytrainLS)

plt.title("Combined Decision Boundary for Non-Linearly Separable Data")
decision_boundary(NLSclassifier, xtrainNLS, ytrainNLS, density = 0.025)


#%% Q3 Confusion Matrices and Accuracy Scores

print("For Linearly Separable Data\n")
LSout = LSclassifier.predict(xtestLS)
print("Confusion Matrix:")
print(confusion_matrix(ytestLS, LSout))
print("Accuracy Score =", accuracy_score(ytestLS, LSout))


print("\n\nFor Non-Linearly Separable Data\n")
NLSout = NLSclassifier.predict(xtestNLS)
print("Confusion Matrix:")
print(confusion_matrix(ytestNLS, NLSout))
print("Accuracy Score =", accuracy_score(ytestNLS, NLSout))

