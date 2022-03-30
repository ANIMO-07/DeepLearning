
#%%
from cgi import print_arguments
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join
import math
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score

# Colecting LS Data

LSpath = "/home/abc/Documents/sem 6/deep learning/Group 16_Assignment1/data/Group16/Classification/LS_Group16/"
NLSpath = "/home/abc/Documents/sem 6/deep learning/Group 16_Assignment1/data/Group16/Classification/NLS_Group16.txt"
UNIpath = "/home/abc/Documents/sem 6/deep learning/Group 16_Assignment1/data/Group16/Regression/UnivariateData/16.csv"
classes12 = ["Class1.txt", "Class2.txt"]
classes23 = ["Class2.txt", "Class3.txt"]
classes13 = ["Class1.txt", "Class3.txt"]

classes = ["Class1.txt", "Class2.txt", "Class3.txt"]

def give_data():
    dfs = []
    df = pd.read_csv(UNIpath,names=["x","y"],sep=",",dtype={
        "x":"double","y":"double"},engine="python")
    # print(df)
    dfs.append(df)
    ls_data = pd.concat(dfs,axis=0).reset_index(drop=True)

    train,validation,test = np.split(ls_data.sample(
        frac=1), [int(.6*len(ls_data)), int(.8*len(ls_data))])

    return train,validation,test


trainer,validation,test = give_data()

def sigmoid(an):
    return 1/(1+math.exp(-an))


def give_error(yn, sn):
    En = math.pow((yn-sn), 2)/2
    return En


def differentiate(an):
    diff = sigmoid(an)*(1-sigmoid(an))
    return diff

def delta_output(ynk, ynk_, ank):
    tmp = (ynk-ynk_)*differentiate(ank)
    return tmp

def delta_input(delnkList, ank, wts):
    tmp = 0
    for k in range(2):
        tmp += delnkList[k]*wts[k]
    tmp *= differentiate(ank)
    return tmp

def inst_error(ynk, ynk_):
    tmp = math.pow((ynk-ynk_), 2)
    return tmp

def average_error(errorArray):
    if(len(errorArray) == 0):
        return 10000000
    avg_err = 0
    for err in errorArray:
        avg_err += err
    avg_err /= len(errorArray)

    return avg_err

def mlpTest(test,wij,wjk,bias_inputs,bias_output,n_input,n_hidden,n_output):
    eta = 0.01
    predicted_class = []

    for n in range(len(test)):
         xn = [test.iloc[n,0]]
         anj = []
         for j in range(n_hidden):
             tmp = 0
             for i in range(n_input):
                 tmp += wij[i][j]*xn[i]
             tmp += bias_inputs[j]
             anj.append(tmp)
        
         ank = []
         for k in range(n_output):
             tmp = 0
             for j in range(n_hidden):
                 tmp += wjk[j][k]*(sigmoid(anj[j]))
             tmp += bias_output[k]
             ank.append(tmp)
        
         predicted_class.append(ank)
    
    return predicted_class

def mlpTest2(test,wij,wjk,bias_inputs,bias_output,n_input,n_hidden,n_output):
    eta = 0.01
    predicted_class = []
    anj_array = []

    for n in range(len(test)):
         xn = [test.iloc[n,0]]
         anj = []
         for j in range(n_hidden):
             tmp = 0
             for i in range(n_input):
                 tmp += wij[i][j]*xn[i]
             tmp += bias_inputs[j]
             anj.append(tmp)
        
         ank = []
         for k in range(n_output):
             tmp = 0
             for j in range(n_hidden):
                 tmp += wjk[j][k]*(sigmoid(anj[j]))
             tmp += bias_output[k]
             ank.append(tmp)
        
         predicted_class.append(ank)
         anj_array.append(anj)

    
    return predicted_class,anj_array

def mlpTrain(train,n_hidden,n_output,n_input):

    eta = 0.01
    # input weights
    wij = []
    for i in range(n_input):
        tmp = []
        for j in range(n_hidden):
            tmp.append(random.random())
        wij.append(tmp)
    
    wjk = []
    for j in range(n_hidden):
        tmp = []
        for k in range(n_output):
            tmp.append(random.random())
        wjk.append(tmp)
    
    bias_input = []
    for j in range(n_hidden):
        bias_input.append(random.random())

    bias_output = []
    for k in range(n_output):
        bias_output.append(random.random())
    prev_error = 1000
    error = 1000
    graph = []
    while(error>0.01):
        print(error)
        prev_error = error
        for n in range(len(train)):
            xn = [train.iloc[n,0]]
            anj = []
            for j in range(n_hidden):
                anj.append(0)
            for j in range(n_hidden):
                for i in range(n_input):
                    anj[j] += xn[i]*wij[i][j]
                anj[j] += bias_input[j]

            ank = []
            for k in range(n_output):
                ank.append(0)

            for k in range(n_output):
                for j in range(n_hidden):
                    ank[k] += sigmoid(anj[j])*wjk[j][k]
                ank[k] += bias_output[k]

            delnk_o = []
            ynk = []
            del_wjk = []

            for k in range(n_output):
                ynk.append(train.iloc[n,1])
                
            for k in range(n_output):
                ynk_ = ank[k]
                delank = 1
                y_diff = ynk[k]-ynk_
                delnk_o.append(y_diff*delank)
            
            for j in range(n_hidden):
                tmp = []
                for k in range(n_output):
                    tmp.append(eta*delnk_o[k]*sigmoid(anj[j]))
                del_wjk.append(tmp)

            for j in range(n_hidden):
                for k in range(n_output):
                    wjk[j][k] += del_wjk[j][k]
                    bias_output[k] += eta*delnk_o[k]
            
            delnj = []

            for i in range(n_input):
                for j in range(n_hidden):
                    tmp = 0
                    for k in range(n_output):
                        tmp += delnk_o[k]*wjk[j][k]*differentiate(anj[j])
                    wij[i][j] += tmp*eta*xn[i]
                    bias_input[j] += eta*tmp
        
        error_array = []
        predicted_class = mlpTest(trainer,wij,wjk,bias_input,bias_output,n_input,n_hidden,n_output)
        for n in range(len(trainer)):
            error_array.append(0)

            for k in range(n_output):
                error_array[n] += inst_error(trainer.iloc[n,1], predicted_class[n][k])
            
            error_array[n] = error_array[n]/2

        error = average_error(error_array)
        graph.append(error)

    return wij,bias_input,wjk,bias_output, graph

wij,bias_input,wjk,bias_output,graph = mlpTrain(trainer, 4, 1, 1)

epoch = []

for i in range(len(graph)):
    epoch.append(i)
    
plt.plot(epoch, graph)
plt.show()

test_output = mlpTest(test, wij, wjk, bias_input, bias_output, 1, 4, 1)

for i in range(len(test)):
    print(test.iloc[i, 1], test_output[i])

#Q2
def mse(data):
    predicted_value = mlpTest(data, wij, wjk, bias_input, bias_output, 1, 4, 1)
    return mean_squared_error(data.iloc[:, 1], predicted_value)

mse_train = mse(trainer)
mse_test = mse(test)
mse_validation = mse(validation)

plt.bar(["train", "test", "validation"],[mse_train, mse_test, mse_validation])
plt.show()

#Q3
predicted_value = mlpTest(trainer, wij, wjk, bias_input, bias_output, 1, 4, 1)
plt.scatter(trainer.iloc[:, 0], trainer.iloc[:, 1], c = "b", label = "Target Output")
plt.scatter(trainer.iloc[:, 0], predicted_value, c = "r", label = "Model Output")
plt.xlabel("input variable")
plt.ylabel("output variable")
plt.title("Training data")
plt.legend()
plt.show()

predicted_value = mlpTest(test, wij, wjk, bias_input, bias_output, 1, 4, 1)
plt.scatter(test.iloc[:, 0], test.iloc[:, 1], c = "b", label = "Target Output")
plt.scatter(test.iloc[:, 0], predicted_value, c = "r", label = "Model Output")
plt.xlabel("input variable")
plt.ylabel("output variable")
plt.title("Test data")
plt.legend()
plt.show()

predicted_value = mlpTest(validation, wij, wjk, bias_input, bias_output, 1, 4, 1)
plt.scatter(validation.iloc[:, 0], validation.iloc[:, 1], c = "b", label = "Target Output")
plt.scatter(validation.iloc[:, 0], predicted_value, c = "r", label = "Model Output")
plt.xlabel("input variable")
plt.ylabel("output variable")
plt.title("Validation data")
plt.legend()
plt.show()

# Q4
predicted_value = mlpTest(trainer, wij, wjk, bias_input, bias_output, 1, 4, 1)
plt.scatter(trainer.iloc[:, 1], predicted_value)
plt.xlabel("Actual value of output variable")
plt.ylabel("Predicted value of output variable")
plt.title("Training data")
plt.show()

predicted_value = mlpTest(test, wij, wjk, bias_input, bias_output, 1, 4, 1)
plt.scatter(test.iloc[:, 1], predicted_value)
plt.xlabel("Actual value of output variable")
plt.ylabel("Predicted value of output variable")
plt.title("Test data")
plt.show()

predicted_value = mlpTest(validation, wij, wjk, bias_input, bias_output, 1, 4, 1)
plt.scatter(validation.iloc[:, 1], predicted_value)
plt.xlabel("Actual value of output variable")
plt.ylabel("Predicted value of output variable")
plt.title("Validation data")
plt.show()

#%%

def neuronOutputs(data, x):
    
    for i in range(x):
        neuron = []
        for lst in data:
            neuron.append(lst[i])

        clr = ["r", "g", "b"]
        
        plt.scatter(test.iloc[:,0], neuron)

        plt.show()


test_output,hidden_output = mlpTest2(test, wij, wjk, bias_input, bias_output, 1, 4, 1)

neuronOutputs(hidden_output,4)
test_output = mlpTest(test, wij, wjk, bias_input, bias_output, 1, 4, 1)

    

# %%
