# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join
import math
import random
from sklearn.metrics import confusion_matrix, accuracy_score

# %%

NLSpath = "/home/abc/Documents/sem 6/deep learning/Group 16_Assignment1/data/Group16/Classification/NLS_Group16.txt"

def seperate_data():
    df2 = pd.read_csv(NLSpath, names=["x", "y"], sep=" ", dtype={
                         "x": "float", "y": "float"}, engine="python",skiprows=1,index_col=False)
    classes = []
    for i in range(1500):
        if(i < 500):
            classes.append(0)
        elif(i < 1000):
            classes.append(1)
        else:
            classes.append(2)
    df2["class"] = classes
    # print(df2)
    dfs = [df2]
    nls_data = pd.concat(dfs, axis=0).reset_index(drop=True)
    # print(nls_data)
    
    train, validation, test = np.split(nls_data.sample(
        frac=1), [int(.6*len(nls_data)), int(.8*len(nls_data))])

    dfx = train.copy()
    dfx.drop(["class"],axis=1,inplace=True)

    return train, validation, test,dfx

trainer,validation,test,dfx = seperate_data()
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
    hidden1 = []
    hidden2 = []

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
         hidden1.append(anj)
         hidden2.append(anl)

    return predicted_class, hidden1, hidden2

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
                if(train.iloc[n,2] == k):
                    ynk.append(1)
                else:
                    ynk.append(0)
                
            for k in range(n_output):
                ynk_ = sigmoid(ank[k])
                delank = differentiate(ank[k])
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
                x = 0
                if(k==trainer.iloc[n,2]):
                    x = 1
                error_array[n] += inst_error(x, sigmoid(predicted_class[n][k]))

        error = average_error(error_array)
        graph.append(error)

    return wij,wjl,wlk,bias_input1,bias_input2,bias_output,graph

wij,wjl,wlk,bias_input1,bias_input2,bias_output, graph = mlpTrain(trainer, 10, 8, 3, 2)

test_output = mlpTest(test, wij, wjl, wlk, bias_input1, bias_input2, bias_output, 2, 10, 8, 3)

epoch = []

for i in range(len(graph)):
    epoch.append(i)

print(graph)
plt.plot(epoch, graph)
plt.show()

# %%

def final_class(test_output, n_output):

    final_class = []
    for i in range(len(test_output)):
        ind_max = 0
        maxi = 0
        for k in range(n_output):
            if(maxi < sigmoid(test_output[i][k])):
                maxi = sigmoid(test_output[i][k])
                ind_max = k

        final_class.append(ind_max)

    return final_class

final_classes = final_class(test_output, 3)

x = 0

print(confusion_matrix(test.iloc[:, 2], final_classes))
print(accuracy_score(test.iloc[:, 2], final_classes))

def decision_boundary(X, y, density = 0.1, padding = 1):
    
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
        grid_tmp = pd.DataFrame(grid.tolist())
        test_output = mlpTest(grid_tmp, wij, wjl, wlk, bias_input1, bias_input2, bias_output, 2, 10, 8, 3)

        yhat = final_class(test_output, 3)
        yhat = np.array(yhat)
        print(yhat)

        zz = yhat.reshape(xx.shape)

        plt.contourf(xx, yy, zz, cmap='Paired')

        plt.scatter(X[:,0], X[:,1], c = y, s = 5)
        plt.show()


decision_boundary(dfx,trainer["class"])

clr = ["r", "g", "b"]
plt.scatter(test.iloc[:, 0], test.iloc[:, 1], color = [clr[k] for k in test.iloc[:, 2]])
plt.show()

clr = ["r", "g", "b"]
plt.scatter(test.iloc[:, 0], test.iloc[:, 1], color = [clr[k] for k in final_classes])
plt.show()
#%%
test_output,hidden_output1,hidden_output2 = mlpTest2(test, wij, wjl, wlk, bias_input1, bias_input2, bias_output, 2, 10, 8, 3)


# %%

def neuronOutputs(data, x):
    
    for i in range(x):
        neuron = []
        for lst in data:
            neuron.append(sigmoid(lst[i]))

        clr = ["r", "g", "b"]
        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")
        ax.scatter(test.iloc[:,0], test.iloc[:,1], neuron, color = [clr[k] for k in test.iloc[:, 2]])

        plt.show()


# neuronOutputs(hidden_output1, 10)
# neuronOutputs(hidden_output2, 8)
neuronOutputs(test_output, 3)
    

# %%
