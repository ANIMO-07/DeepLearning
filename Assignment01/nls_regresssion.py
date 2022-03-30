#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join
import math
import random
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error

NLSpath = "./data/Group16/Classification/NLS_Group16.txt"
BIpath = "./data/Group16/Regression/BivariateData/16.csv"

def give_data():
    dfs = []
    df = pd.read_csv(BIpath,names=["x","y","z"],sep=",",dtype={
        "x":"double","y":"double","z":"double"},engine="python")
    # print(df)
    dfs.append(df)
    ls_data = pd.concat(dfs,axis=0).reset_index(drop=True)

    train,validation,test = np.split(ls_data.sample(
        frac=1), [int(.6*len(ls_data)), int(.8*len(ls_data))])

    dfx = train.copy()
    dfx.drop(["z"],axis=1,inplace=True)

    return train, validation, test,dfx


trainer,validation,test,dfx = give_data()
# print(trainer)

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
    tmp = (math.pow((ynk-ynk_), 2))/2
    return tmp

def average_error(errorArray):
    if(len(errorArray) == 0):
        return 10000000
    avg_err = 0
    for err in errorArray:
        avg_err += err
    avg_err /= len(errorArray)

    return avg_err


def mlpTest(test,wij,wjl,wlk,bias_input1,bias_input2,bias_output,n_input,n_hidden1,n_hidden2,n_output):
    eta = 0.01
    predicted_class = []
    anj_array = []
    anl_array = []

    for n in range(len(test)):
         xn = [test.iloc[n,0],test.iloc[n,1]]
         anj = []
         for j in range(n_hidden1):
             tmp = 0
             for i in range(n_input):
                 tmp += wij[i][j]*xn[i]
             tmp += bias_input1[j]
             anj.append(tmp)
         
         anl = []
         for l in range(n_hidden2):
             tmp = 0
             for j in range(n_hidden1):
                 tmp += wjl[j][l]*(sigmoid(anj[j]))
             tmp += bias_input2[l]
             anl.append(tmp)


         ank = []
         for k in range(n_output):
             tmp = 0
             for l in range(n_hidden2):
                 tmp += wlk[l][k]*(sigmoid(anl[l]))
             tmp += bias_output[k]
             ank.append(tmp)
        
         predicted_class.append(ank)
         

        
    return predicted_class


def mlpTest2(test,wij,wjl,wlk,bias_input1,bias_input2,bias_output,n_input,n_hidden1,n_hidden2,n_output):
    eta = 0.01
    predicted_class = []
    anj_array = []
    anl_array = []

    for n in range(len(test)):
         xn = [test.iloc[n,0],test.iloc[n,1]]
         anj = []
         for j in range(n_hidden1):
             tmp = 0
             for i in range(n_input):
                 tmp += wij[i][j]*xn[i]
             tmp += bias_input1[j]
             anj.append(tmp)
         
         anl = []
         for l in range(n_hidden2):
             tmp = 0
             for j in range(n_hidden1):
                 tmp += wjl[j][l]*(sigmoid(anj[j]))
             tmp += bias_input2[l]
             anl.append(tmp)


         ank = []
         for k in range(n_output):
             tmp = 0
             for l in range(n_hidden2):
                 tmp += wlk[l][k]*(sigmoid(anl[l]))
             tmp += bias_output[k]
             ank.append(tmp)
        
         predicted_class.append(ank)
         anj_array.append(anj)
         anl_array.append(anl)

        
    return predicted_class,anj_array,anl_array


def mlpTrain(train,n_hidden1,n_hidden2,n_output,n_input):

    eta = 0.01
    # input weights
    wij = []
    for i in range(n_input):
        tmp = []
        for j in range(n_hidden1):
            tmp.append(random.random())
        wij.append(tmp)
    
    wjl = []
    for j in range(n_hidden1):
        tmp = []
        for l in range(n_hidden2):
            tmp.append(random.random())
        wjl.append(tmp)
    
    wlk = []

    for l in range(n_hidden2):
        tmp = []
        for k in range(n_output):
            tmp.append(random.random())
        wlk.append(tmp)

    bias_input1 = []
    for j in range(n_hidden1):
        bias_input1.append(random.random())

    bias_input2 = []
    for l in range(n_hidden2):
        bias_input2.append(random.random())

    bias_output = []
    for k in range(n_output):
        bias_output.append(random.random())

    prev_error = 1000
    error = 1000
    graph = []
    while(error>0.01):
        prev_error = error
        print(prev_error, error)
        for n in range(len(train)):
            xn = [train.iloc[n,0],train.iloc[n,1]]
            anj = []
            for j in range(n_hidden1):
                anj.append(0)
            for j in range(n_hidden1):
                for i in range(n_input):
                    anj[j] += xn[i]*wij[i][j]
                anj[j] += bias_input1[j]
            
            anl = []
            for l in range(n_hidden2):
                anl.append(0)
            for l in range(n_hidden2):
                for j in range(n_hidden1):
                    anl[l] += sigmoid(anj[j])*wjl[j][l]
                anl[l] += bias_input2[l]

            ank = []
            for k in range(n_output):
                ank.append(0)

            for k in range(n_output):
                for l in range(n_hidden2):
                    ank[k] += sigmoid(anl[l])*wlk[l][k]
                ank[k] += bias_output[k]

            delnk_o = []
            ynk = []
            del_wlk = []

            for k in range(n_output):
                ynk.append(train.iloc[n,2])
                
            for k in range(n_output):
                ynk_ = ank[k]
                delank = 1
                y_diff = ynk[k]-ynk_
                delnk_o.append(y_diff*delank)
            
            for l in range(n_hidden2):
                tmp = []
                for k in range(n_output):
                    tmp.append(eta*delnk_o[k]*sigmoid(anl[l]))
                del_wlk.append(tmp)

            for l in range(n_hidden2):
                for k in range(n_output):
                    wlk[l][k] += del_wlk[l][k]
                    bias_output[k] += eta*delnk_o[k]
            
            del_wjl = []

            for j in range(n_hidden1):
                for l in range(n_hidden2):
                    tmp = 0
                    for k in range(n_output):
                        tmp += delnk_o[k]*wlk[l][k]*differentiate(anl[l])
                    wjl[j][l] += tmp*eta*sigmoid(anj[j])
                    bias_input2[l] += eta*tmp
                
            for i in range(n_input):
                for j in range(n_hidden1):
                    tmp = 0
                    for k in range(n_output):
                        product = delnk_o[k]
                        l_sum = 0
                        for l in range(n_hidden2):
                            l_sum += wlk[l][k]*differentiate(anl[l])*wjl[j][l]
                        product *= l_sum
                        tmp += product
                    wij[i][j] += tmp*eta*differentiate(anj[j])*xn[i]
                    bias_input1[j] += eta*tmp*differentiate(anj[j])
        
        error_array = []
        predicted_class = mlpTest(trainer,wij,wjl,wlk,bias_input1,bias_input2,bias_output,n_input,n_hidden1,n_hidden2,n_output)
        for n in range(len(trainer)):
            error_array.append(0)
            for k in range(n_output):
                error_array[n] += inst_error(trainer.iloc[n,2], predicted_class[n][k])

        error = average_error(error_array)
        graph.append(error)

    return wij,wjl,wlk,bias_input1,bias_input2,bias_output,graph

wij,wjl,wlk,bias_input1,bias_input2,bias_output, graph = mlpTrain(trainer, 5, 3, 1, 2)

test_output = mlpTest(test, wij, wjl, wlk, bias_input1, bias_input2, bias_output, 2, 5, 3, 1)

epoch = []

for i in range(len(graph)):
    epoch.append(i)

print(graph)
plt.plot(epoch, graph)
plt.show()

for i in range(len(test)):
    print(test.iloc[i, 2], test_output[i])

# #Q2
# def mse(data):
#     predicted_value = mlpTest(test, wij, wjl, wlk, bias_input1, bias_input2, bias_output, 2, 5, 3, 1)
#     return mean_squared_error(data.iloc[:, 2], predicted_value)

# mse_train = mse(trainer)
# mse_test = mse(test)
# mse_validation = mse(validation)

# plt.bar(["train", "test", "validation"],[mse_train, mse_test, mse_validation])
# plt.show()

#Q3
predicted_value = mlpTest(trainer, wij, wjl, wlk, bias_input1, bias_input2, bias_output, 2, 5, 3, 1)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(trainer.iloc[:,0], trainer.iloc[:,1], trainer.iloc[:,2], c = "b", label = "Target Output")
ax.scatter(trainer.iloc[:,0], trainer.iloc[:,1], predicted_value, c = "r", label = "Model Output")
plt.title("Bivariate Train Data")
plt.legend()
plt.show()

predicted_value = mlpTest(test, wij, wjl, wlk, bias_input1, bias_input2, bias_output, 2, 5, 3, 1)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(test.iloc[:,0], test.iloc[:,1], test.iloc[:,2], c = "b", label = "Target Output")
ax.scatter(test.iloc[:,0], test.iloc[:,1], predicted_value, c = "r", label = "Model Output")
plt.title("Bivariate Test Data")
plt.legend()
plt.show()

predicted_value = mlpTest(validation, wij, wjl, wlk, bias_input1, bias_input2, bias_output, 2, 5, 3, 1)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(validation.iloc[:,0], validation.iloc[:,1], validation.iloc[:,2], c = "b", label = "Target Output")
ax.scatter(validation.iloc[:,0], validation.iloc[:,1], predicted_value, c = "r", label = "Model Output")
plt.title("Bivariate Validation Data")
plt.legend()
plt.show()


# Q4

# ########
predicted_value = mlpTest(trainer, wij, wjl, wlk, bias_input1, bias_input2, bias_output, 2, 5, 3, 1)
plt.scatter(trainer.iloc[:, 2], predicted_value)
plt.xlabel("Actual value of output variable")
plt.ylabel("Predicted value of output variable")
plt.title("Training data")
plt.show()




predicted_value = mlpTest(test, wij, wjl, wlk, bias_input1, bias_input2, bias_output, 2, 5, 3, 1)
plt.scatter(test.iloc[:, 2], predicted_value)
plt.xlabel("Actual value of output variable")
plt.ylabel("Predicted value of output variable")
plt.title("Test data")
plt.show()


predicted_value = mlpTest(validation, wij, wjl, wlk, bias_input1, bias_input2, bias_output, 2, 5, 3, 1)
plt.scatter(validation.iloc[:, 2], predicted_value)
plt.xlabel("Actual value of output variable")
plt.ylabel("Predicted value of output variable")
plt.title("Validation data")
plt.show()




#%%
def neuronOutputs(data, x):
    
    for i in range(x):
        neuron = []
        for lst in data:
            neuron.append(sigmoid(lst[i]))

        clr = ["r", "g", "b"]
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        ax.scatter(test.iloc[:,0], test.iloc[:,1], neuron)

        plt.show()

test_output,hidden1,hidden2 = mlpTest2(test, wij, wjl, wlk, bias_input1, bias_input2, bias_output, 2, 5, 3, 1)


neuronOutputs(test_output,1)
neuronOutputs(hidden1,5)
neuronOutputs(hidden2,3)

# %%
