# -*- coding: utf-8 -*-

#from theano import *
#import theano.tensor as T
import numpy as np
#import scipy as sp
#from functools import partial
from sklearn.neural_network import MLPClassifier
#from sklearn.preprocessing import StandardScaler #Scaler


def importtrain(fname):
    Z = np.loadtxt(fname, delimiter=',', skiprows=1)
    n,m = np.add(np.shape(Z), -1)
    X = np.c_[np.ones(n+1), Z[:,1:m]]          # 1st columns to ones, skip first column since it is client number
    y = Z[:,m]                          # extract "y" column
    index = Z[:,0]                      # client number
    theta = np.zeros(m)                 # intialize theta
    X = np.c_[X, X[:,1:m] * X[:,1:m]]  #feature matrix + append cuadratic features
    return X, y, index, theta

def importtest(ftest):
    Z = np.loadtxt(ftest, delimiter=',', skiprows=1)
    n,m = np.shape(Z)
    X = np.c_[np.ones(n), Z[:,1:(m)]]   # 1st columns to ones, skip first column since it is client number
    index = Z[:,0]                      # client number
    theta = np.zeros(m)
    X = np.c_[X, X[:,1:m] * X[:,1:m]]   #feature matrix + append cuadratic features
    return X, y, index, theta

def cleanfeaturescaling(X, Z):    # X must be training set, Z must be test set
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    for r in reversed(range(1, np.shape(X)[1])):    # remove columns with sigma = 0, starting from the right to the left.
        if sigma[r] == 0:
            X = np.delete(X, r, 1)
            Z = np.delete(Z, r, 1)
    mu = np.mean(X, axis=0)          #recalculate mu and sigma after removing columns
    sigma = np.std(X, axis=0)
    mu1 = np.mean(Z, axis=0)
    sigma1 = np.std(Z, axis=0)
    sigma1[sigma1 == 0] = 1           # to prevent dividing by 0
    sigma[sigma == 0] = 1             # to prevent dividing by 0
    X = np.divide(np.add(X, - mu), sigma)   # feature scaling X
    Z = np.divide(np.add(Z, - mu1), sigma1) # feature scaling Z
    X[:,0] = 1        # assign 1st column to ones
    Z[:,0] = 1      # assign 1st column to ones
    return X, Z
    
def sigmoid(X, theta):
    h = 1/(1+np.exp(np.matmul(X,theta)))    # calculate element-wise sigmoid function
    return h

def costfunction(X, y, theta):
    m = float(len(X))       #calculate cost and gradient
    J = np.multiply(1/m, (np.matmul(-y,np.log(sigmoid(X,theta))) - np.matmul((1-y), np.log(1-sigmoid(X,theta)))))
    grad = np.multiply(1/m, (np.matmul(np.transpose(X), (sigmoid(X,theta)-y))))
    return J, grad
    
def normalequation(X, y):
    l = np.shape(X)[1]     #calculate normal equation to return theta directly 
    r_lambda =  100
    theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X), X)+ r_lambda * np.identity(l)), np.transpose(X)), y)
    return theta
    
def neuralnetwork(X, Z, y):     # neural network MLPC clasifier 370, 200 ----> 0,81
    clf = MLPClassifier(hidden_layer_sizes=(640, 640, 370, 200, 100, 50), activation='relu', beta_1=0.9, beta_2=0.999, learning_rate_init=0.001, learning_rate='adaptive', early_stopping = True, shuffle = True, warm_start = True, validation_fraction = 0.3, max_iter = 25000, random_state = 1235)
    clf.fit(X, y)
    y_pred = clf.predict_proba(Z)[:,1]
    return y_pred


X, y, index, theta = importtrain("train.csv")   # call to import training archive

Z , y, index2, theta2 = importtest("test.csv")   # call to import test archive

X, Z = cleanfeaturescaling(X, Z)              # feature scaling of training and test sets

#theta = normalequation(X, y)                # theta using normal equation for logistic regresion

#sol = sigmoid(Z, theta)                      # test theta parameters

#np.savetxt("foo.csv",sol, delimiter=",")  # save to csv

y_sol = neuralnetwork(X, Z, y)              # neural networks output

np.savetxt("foonn.csv", y_sol, delimiter=",")  # save to csv
