# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:56:10 2021
@author: Abhishek Ramesh
Stochastic Gradient Descent on MNSIT Dataset
"""

#%% Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#%% Functions
#Converts real number vector into probability vector
def softmax_multiclass(vector):
    exp_vector = np.exp(vector)
    return exp_vector/(np.sum(exp_vector))

#Finds how closely related the groundtruth and the predictive probabilities are
def crossentropy_multiclass(ground_truth, predictive):
    return -np.vdot(ground_truth, np.log(predictive))


#Gradient Loss Function
def grad_L(beta, X, y):
    N = X.shape[0]
    gradloss = 0
    
    for i in range(N):
        xiHat = X[i]
        yi = y[i]
        predictive_i = softmax_multiclass(beta@xiHat)
        
        grad_i = crossentropy_multiclass(yi, predictive_i)
        
        gradloss += grad_i
    
    return gradloss

#Computing L to find lowest Objective Function Value per Iteration
def minimizeL(X, y, alpha, batch_size, numEpochs):
    N, d = X.shape
    X = np.insert(X, 0, 1, axis = 1)
    K = y.shape[1]
    beta = np.zeros((K, d+1))
    
    Lvals = []
    
    for ep in range(numEpochs):
        L = grad_L(beta, X, y)
        Lvals.append(L)
        
        print("Epoch is: " + str(ep) + " Cost is: " + str(L))
               
        prm = np.random.permutation(N)
        
        batch_idx = 0
        for start_idx in range(0, N, batch_size):
            stop_idx = start_idx + batch_size
            stop_idx = min(stop_idx, N)
            
            num_examples_in_batch = stop_idx - start_idx
            
            grad_Li = np.zeros(d+1)
        
            for i in prm[start_idx:stop_idx]:
                xiHat = X[i]
                Yi = y[i]
                
                #Compute gradient and update beta                
                predictive_i = softmax_multiclass(beta @ xiHat)
                grad_Li = np.outer(predictive_i - Yi, xiHat)                   
            
            
            grad_Li = grad_Li/num_examples_in_batch
            beta = beta - alpha*grad_Li
            batch_idx += 1
                    
    return beta, Lvals

#Predict number value for other images
def predictLabels(X, beta):
    X = np.insert(X, 0, 1, axis = 1)
    N = X.shape[0]
    
    predictions = []
    probabilities = []
    
    for i in range(N):
        xiHat = X[i]
        predictive_i = softmax_multiclass(beta @ xiHat)
        
        k = np.argmax(predictive_i)
        predictions.append(k)
        
        probabilities.append(np.max(predictive_i))
        
    return predictions, probabilities


#%% Accuracy of values
def accuracy(ground_truth, prediction):
    numCorrect = 0
    numTest = ground_truth.shape[0]
    
    for i in range(numTest):
        if predictions[i] == ground_truth[i]:
            numCorrect += 1
    
    accuracy = (numCorrect/numTest)*100
    
    return (accuracy)
      
#%% Pre-process data
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train/255.0
X_test = X_test/255.0

N_train, numRows, numCols = X_train.shape
X_train = np.reshape(X_train, (N_train, numRows*numCols))

Y_train = pd.get_dummies(Y_train).values

#%% Multiclass Logistic Regression used on MNIST Dataset
np.random.seed(3)

alpha = 0.01
numEpochs = 10
batch_size = 5
beta, Lvals = minimizeL(X_train, Y_train, alpha, batch_size, numEpochs)

plt.semilogy(Lvals)
plt.xlabel("Iterations")
plt.ylabel("Objective Function Value")

#Now test on data
N_test = X_test.shape[0]
X_test = np.reshape(X_test, (N_test, numRows*numCols))
predictions, probabilities = predictLabels(X_test, beta)

print("Accuracy of function is: " + str(accuracy(Y_test, predictions)) + "%")

#%% Difficult Examples
probabilities = np.array(probabilities)
agreement = (predictions == Y_test)
sortedindexs = np.argsort(probabilities)
sortedindexs = sortedindexs[::-1]

difficult_examples = []

for i in sortedindexs:
    if (agreement[i] == False):
        difficult_examples.append(i)
        
i = difficult_examples[2]
Xi = np.reshape(X_test[i], (28,28))
plt.imshow(Xi)

print("Predicted Value: " + str(predictions[i]))
print("Ground Truth Value: " + str(Y_test[i]))
print("Probability of accuracy: " + str(probabilities[i]))