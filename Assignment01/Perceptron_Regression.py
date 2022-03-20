# %% Imports

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split




# %% Collecting Univariate Data

unidata = pd.read_csv("./data/Group16/Regression/UnivariateData/16.csv", names = ["x", "y"], 
					dtype = {"x" : "float", "y" : "float"}, engine = "python", index_col = False)

# %% Collecting Bivariate Data

bidata = pd.read_csv("./data/Group16/Regression/BivariateData/16.csv", names = ["x", "y", "z"], 
					dtype = {"x" : "float", "y" : "float", "z" : "float"}, engine = "python", index_col = False)
# %%

class perceptron:
	def __init__(self):
		self.weights = []
		self.errorvsepochs = []
 
	def _give_error(self, yn, sn):
		En = math.pow((yn-sn),2)/2
		return En

	def fit(self, X, y, initial_vec_val = 0,  epochs=30, eta = 0.4, thresh = 1E-3):

		dimensions = len(X.columns)

		weights = [initial_vec_val] * (dimensions + 1)

		x = X.to_numpy()
		y = y.to_numpy()

		N = len(x)

		inputx = np.c_[np.ones(N), x]

		errvsepochs = []

		for i in range(epochs):
			total_epoch_error = 0

			for j in range(N):
				predicted_output = sum(np.multiply(weights, inputx[j]))

				true_output = y[j]

				instantaneous_error = self._give_error(true_output, predicted_output)
				total_epoch_error += instantaneous_error

				delta_w = eta * (true_output-predicted_output) * inputx[j]
	
				# print(i, j, predicted_output, true_output, instantaneous_error, delta_w)
				weights = np.add(weights, delta_w)
			
			errvsepochs.append(total_epoch_error/N)

			if i > 0:
				if errvsepochs[i-1] - errvsepochs[i] < thresh:
					break
			# print(wts)
			# print("For Epoch:", i, "     Calculated Average Error =", total_epoch_error/N)

		self.errorvsepochs = errvsepochs
		self.weights = weights

		print("Model Fitting Complete")
		
	def geterrs(self):
		return self.errorvsepochs

	def predict(self, Xvals):
		predictions = []

		xvals_ = np.c_[np.ones(len(Xvals)), Xvals]
		weights = self.weights

		for x in xvals_:
			predicted_output = sum(np.multiply(weights, x))
			
			predictions.append(predicted_output)
		
		return np.array(predictions)


# %% Q0

#For Univariate Data

xtrain1, xtest1, ytrain1, ytest1 = train_test_split(unidata.drop(["y"], axis = 1), unidata["y"], test_size = 0.4, random_state = 1)
			# 60-40 split since validation was supposed to be considered as more test data
regressor1 = perceptron()
regressor1.fit(xtrain1, ytrain1, epochs = 10, eta = 0.4, thresh = 1E-4)


#For Bivariate Data

xtrain2, xtest2, ytrain2, ytest2 = train_test_split(bidata.drop(["z"], axis = 1), bidata["z"], test_size = 0.4, random_state = 1)
			# 60-40 split since validation was supposed to be considered as more test data
regressor2 = perceptron()
regressor2.fit(xtrain2, ytrain2, epochs = 10, eta = 0.4, thresh = 1E-4)



# %% Q1

#For Univariate Data

errs1 = regressor1.geterrs()
plt.plot(range(1, len(errs1) + 1), errs1)
plt.xlabel("Epochs")
plt.ylabel("Average Error")
plt.title("Perceptrons for Univariate Data")
plt.show()


#For Bivariate Data

errs2 = regressor2.geterrs()
plt.plot(range(1, len(errs2) + 1), errs2)
plt.xlabel("Epochs")
plt.ylabel("Average Error")
plt.title("Perceptrons for Bivariate Data")
plt.show()



# %% Q2

# On Univariate Data

trainout1 = regressor1.predict(xtrain1)
msetrain1 = mean_squared_error(trainout1, ytrain1)

testout1 = regressor1.predict(xtest1)
msetest1 = mean_squared_error(testout1, ytest1)

plt.bar(["Train", "Test"], [msetrain1, msetest1])
plt.xlabel("Data")
plt.ylabel("Mean Squared Error")
plt.title("On Univariate Data")
plt.show()



# On Bivariate Data

trainout2 = regressor2.predict(xtrain2)
msetrain2 = mean_squared_error(trainout2, ytrain2)

testout2 = regressor2.predict(xtest2)
msetest2 = mean_squared_error(testout2, ytest2)

plt.bar(["Train", "Test"], [msetrain2, msetest2])
plt.xlabel("Data")
plt.ylabel("Mean Squared Error")
plt.title("On Bivariate Data")
plt.show()

# %% Q3

# %% Univariate Data

plt.scatter(xtrain1, ytrain1, c = "b", label = "Target Output")
plt.scatter(xtrain1, trainout1, c = "r", label = "Model Output")
plt.legend()
plt.title("Univariate Train Data")
plt.show()

plt.scatter(xtest1, ytest1, c = "b", label = "Target Output")
plt.scatter(xtest1, testout1, c = "r", label = "Model Output")
plt.legend()
plt.title("Univariate Test Data")
plt.show()

# %% Bivariate Data

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(xtrain2["x"], xtrain2["y"], ytrain2, c = "b", label = "Target Output")
ax.scatter(xtrain2["x"], xtrain2["y"], trainout2, c = "r", label = "Model Output")
plt.title("Bivariate Train Data")
plt.legend()
plt.show()


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(xtest2["x"], xtest2["y"], ytest2, c = "b", label = "Target Output")
ax.scatter(xtest2["x"], xtest2["y"], testout2, c = "r", label = "Model Output")
plt.title("Bivariate Test Data")
plt.legend()
plt.show()



# %% Q4

plt.scatter(ytrain1, trainout1)
plt.xlabel("Target Output")
plt.ylabel("Model Output")
plt.title("Univariate Train Data")
plt.show()

plt.scatter(ytest1, testout1)
plt.xlabel("Target Output")
plt.ylabel("Model Output")
plt.title("Univariate Test Data")
plt.show()


# %%

plt.scatter(ytrain2, trainout2)
plt.xlabel("Target Output")
plt.ylabel("Model Output")
plt.title("Bivariate Train Data")
plt.show()

plt.scatter(ytest2, testout2)
plt.xlabel("Target Output")
plt.ylabel("Model Output")
plt.title("Bivariate Test Data")
plt.show()
# %%
