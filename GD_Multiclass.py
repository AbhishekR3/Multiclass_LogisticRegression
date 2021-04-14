# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 22:05:50 2021
@author: Abhishek Ramesh
Gradient Descent on Iris Dataset
"""
#%% Import Libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


#%% Load and Preprocess data
iris = load_iris()
X = iris.data
y = iris.target

#Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=3)

#Insert 1 in the first column to augment the matrix
X_train = np.insert(X_train, 0, 1, axis = 1)
X_t = np.insert(X_test, 0, 1, axis = 1)

#%% Functions
#Sigmoid Multiclass Function
def softmax_multiclass(vector):
    exp_vector = np.exp(vector)
    return exp_vector/(np.sum(exp_vector))

#crossEntropy Loss Function for Multiclass
def crossentropy_multiclass(ground_truth, predictive):
    return -np.vdot(ground_truth, np.log(softmax_multiclass(predictive)))

#Gradient Loss Function
def grad_L(beta, X, y, C):
    N = X.shape[0]
    grad = np.zeros((C, 5))
    
    for i in range(N):
        xiHat = X[[i]]
        yi = np.zeros((C,1))
        yi[y[i]] = 1
        predictive_i = softmax_multiclass(beta @ xiHat.transpose())
        
        grad_i = (predictive_i - yi) @ xiHat
        
        grad += grad_i
    
    return grad/N

#Computing L to find lowest Objective Function Value per Iteration
def minimizeL(X, y, alpha, iterations):
    C = 3
    N, d_1 = X.shape
    
    L_vals = np.zeros(iterations)
    beta_t = np.zeros((C, d_1))
        
    for t in range(iterations):
        grad_i = 0
        
        for i in range(N):
            xiHat = X[[i]].transpose()
            yi = np.zeros((C,1))
            yi[y[i]] = 1
            predictive_i = (beta_t@xiHat)
            grad_i += crossentropy_multiclass(yi, predictive_i)
            
        L_vals[t] = grad_i
        
        print("Iteration: ", t, "Objective Function value: ", L_vals[t])
        
        beta_t = beta_t - alpha*grad_L(beta_t, X, y, C)
    
    return beta_t, L_vals

#%% Gradient Descent on Iris Dataset
np.random.seed(3)
alpha = 1
iterations = 106
beta_t, L_vals = minimizeL(X_train, y_train, alpha, iterations)

plt.plot(L_vals)
plt.xlabel("Iterations")
plt.ylabel("Objective Function Value")

#%% Test Prediction Function
N, d_1 = X_t.shape
C = 3
total = 0

for i in range(N):
    xiHat = X_t[[i]]
    yi = np.zeros((C,1))
    yi[y_test[i]] = 1
    
    predictive_i = softmax_multiclass(beta_t @ xiHat.transpose())
    max_prob = predictive_i.argmax()
    
    test_pred = np.zeros((C,1))                   
    test_pred[max_prob] = 1
    
    check = np.vdot(test_pred, yi)
    
    total += check

accuracy = (total/N)*100
    
print("Accuracy of Gradient Descent function is:", accuracy , "%")