# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join
import math
import random
from sklearn.metrics import confusion_matrix, accuracy_score

# Colecting LS Data

LSpath = "./data/Group16/Classification/LS_Group16/"
NLSpath = "./data/Group16/Classification/NLS_Group16.txt"
classes12 = ["Class1.txt", "Class2.txt"]
classes23 = ["Class2.txt", "Class3.txt"]
classes13 = ["Class1.txt", "Class3.txt"]

classes = ["Class1.txt", "Class2.txt", "Class3.txt"]

def give_data(classes):
    dfs = []
    for i, c in enumerate(classes):
        df = pd.read_csv(join(LSpath, c), names=["x", "y"], sep=" ", dtype={
                         "x": "float", "y": "float"}, engine="python")
        
        df["class"] = i
        
        dfs.append(df)

    ls_data = pd.concat(dfs, axis=0).reset_index(drop=True)

    train, validation, test = np.split(ls_data.sample(
        frac=1), [int(.6*len(ls_data)), int(.8*len(ls_data))])
    dfx = train.copy()
    dfx.drop(["class"],axis=1,inplace=True)

    return train, validation, test,dfx

trainer, validation, test,dfx = give_data(classes)

def sigmoid(an):
    return 1/(1+math.exp(-an))

def give_error(yn, sn):
    En = (math.pow((yn-sn), 2))/2
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
    tmp = math.pow((ynk-ynk_), 2)/2
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
    anj_array = []
    for n in range(len(test)):
         xn = [test.iloc[n,0],test.iloc[n,1]]
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
        
         anj_array.append(anj)
         predicted_class.append(ank)
        
    return anj_array,predicted_class

def mlpTrain(train,n_hidden,n_output,n_input):
    graph = []
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
    error = 0
    count = 0
    while(abs(prev_error-error)>0.001):
        count+=1
        # print(prev_error, error)
        prev_error = error
        for n in range(len(train)):
            xn = [train.iloc[n,0],train.iloc[n,1]]
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
                if(train.iloc[n,2] == k):
                    ynk.append(1)
                else:
                    ynk.append(0)
                
            for k in range(n_output):
                ynk_ = sigmoid(ank[k])
                delank = differentiate(ank[k])
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
        
        # error_array = []
        # predicted_class = mlpTest(validation,wij,wjk,bias_input,bias_output,n_input,n_hidden,n_output)
        # for n in range(len(validation)):
        #     error_array.append(0)

        #     for k in range(n_output):
        #         error_array[n] += inst_error(validation.iloc[n,2], sigmoid(predicted_class[n][k]))
            
        #     error_array[n] = error_array[n]/2

        # error = average_error(error_array)

        error_array = 0
        hidden_output,predicted_class = mlpTest(trainer,wij,wjk,bias_input,bias_output,n_input,n_hidden,n_output)
        for n in range(len(trainer)):
            for k in range(n_output):
                x = 0
                if(k==trainer.iloc[n,2]):
                    x = 1
                error_array += inst_error(x, sigmoid(predicted_class[n][k]))

        error = error_array/len(trainer)
        graph.append(error)

    return wij,bias_input,wjk,bias_output,graph

wij,bias_input,wjk,bias_output,graph = mlpTrain(trainer, 3, 3, 2)

epoch = []

for i in range(len(graph)):
    epoch.append(i)

# print(graph)
plt.plot(epoch, graph)
plt.show()

hidden_output,test_output = mlpTest(test, wij, wjk, bias_input, bias_output, 2, 3, 3)


def neuronOutputs(data):
    neuron1 = []
    neuron2 = []
    neuron3 = []


    for lst in data:
        neuron1.append(sigmoid(lst[0]))
        neuron2.append(sigmoid(lst[1]))
        neuron3.append(sigmoid(lst[2]))


    clr = ["r", "g", "b"]
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter(test.iloc[:,0], test.iloc[:,1], neuron1, color = [clr[k] for k in test.iloc[:, 2]])
    plt.show()

    clr = ["r", "g", "b"]
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter(test.iloc[:,0], test.iloc[:,1], neuron2, color = [clr[k] for k in test.iloc[:, 2]])
    plt.show()

    clr = ["r", "g", "b"]
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter(test.iloc[:,0], test.iloc[:,1], neuron3, color = [clr[k] for k in test.iloc[:, 2]])
    plt.show()


neuronOutputs(hidden_output)
neuronOutputs(test_output)


def final_class(test_output, n_output):

    final_class = []
    for i in range(len(test_output)):
        ind_max = 0
        maxi = 0
        for k in range(n_output):
            if(maxi <= sigmoid(test_output[i][k])):
                maxi = sigmoid(test_output[i][k])
                ind_max = k

        final_class.append(ind_max)

    return final_class

final_classes = final_class(test_output, 3)

print(confusion_matrix(test.iloc[:, 2], final_classes))
print(accuracy_score(test.iloc[:, 2], final_classes))

# for i in range(len(test)):
#     print(test.iloc[i, 2], final_class[i])

clr = ["r", "g", "b"]
plt.scatter(test.iloc[:, 0], test.iloc[:, 1], color = [clr[k] for k in test.iloc[:, 2]])
plt.show()

clr = ["r", "g", "b"]
plt.scatter(test.iloc[:, 0], test.iloc[:, 1], color = [clr[k] for k in final_classes])
plt.show()


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
        hidden_output,test_output = mlpTest(grid_tmp, wij, wjk, bias_input, bias_output, 2, 3, 3)

        yhat = final_class(test_output, 3)
        yhat = np.array(yhat)
        # print(yhat)

        zz = yhat.reshape(xx.shape)

        plt.contourf(xx, yy, zz, cmap='Paired')

        plt.scatter(X[:,0], X[:,1], c = y, s = 5)
        plt.show()


decision_boundary(dfx,trainer["class"])