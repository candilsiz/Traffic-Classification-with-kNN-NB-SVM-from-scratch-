#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:23:30 2023

@author: candilsiz
"""

import numpy as np
from sklearn.base import BaseEstimator


class SupportVectorMachineClassifier(BaseEstimator):
    
    def __init__(self, C=10, features=7, sigma_sq=0.01, kernel="gaussian", n_classes=4):
        self.C = C
        self.features = features
        self.sigma_sq = sigma_sq
        self.kernel = kernel
        self.n_classes = n_classes
        self.weights = [np.zeros(features) for _ in range(n_classes)]
        self.biases = [0. for _ in range(n_classes)]
        
    def __similarity(self, x, l):
        return np.exp(-np.sum((x - l) ** 2) / (2 * self.sigma_sq))
    
    # Apply Gaussian Kernel to data 
    def gaussian_kernel(self, x1, x):
        m = x.shape[0]
        n = x1.shape[0]
        op = [[self.__similarity(x1[x_index], x[l_index]) for l_index in range(m)] for x_index in range(n)]
        return np.array(op)
    
    def hinge_loss_function(self, y, y_hat):
        sum_terms = 1 - y * y_hat
        sum_terms = np.where(sum_terms < 0, 0, sum_terms)
        squared_weights_sum = sum([np.sum(w ** 2) for w in self.weights])
        return (self.C * np.sum(sum_terms) / len(y) + squared_weights_sum / 2)
    
    
    def fit(self, featureTrain, labelTrain, epochs=100, learning_rate=0.01):
        labelTrain = labelTrain.copy()
        featureTrain = featureTrain.copy()
        self.initial = featureTrain.copy()
        
        if self.kernel == "gaussian":
            featureTrain = self.gaussian_kernel(featureTrain, featureTrain)
            
        # Train seperate Binary Classifier for each Label Class with OvR Strategy    
        for i in range(self.n_classes):
            y = np.where(labelTrain == i, 1, -1)
            self.weights[i], self.biases[i] = self.__train_one_vs_rest(featureTrain, y, epochs, learning_rate, i)
            
    def __train_one_vs_rest(self, x, y, epochs, learning_rate, i):
        
        weights = np.zeros(x.shape[0])
        bias = 0
        
        # Gradient Descent, optimize weights and biases for each Class Label
        for epoch in range(epochs):
            y_hat = np.dot(x, weights) + bias
            grad_weights = (-self.C * np.multiply(y, x.T).T + weights).T
            
            for weight in range(weights.shape[0]):
                grad_weights[weight] = np.where(1 - y_hat <= 0, weights[weight], grad_weights[weight])
            
            grad_weights = np.sum(grad_weights, axis=1)
            weights -= learning_rate * grad_weights / x.shape[0]
            grad_bias = -y * bias
            grad_bias = np.where(1 - y_hat <= 0, 0, grad_bias)
            grad_bias = np.sum(grad_bias)
            bias -= grad_bias * learning_rate / x.shape[0]
            
        hinge_loss = self.hinge_loss_function(y, y_hat)
        print(f"Hinge Loss for Class {i} is: {hinge_loss}")
        
        return weights, bias 

    def predict(self, x):
        if self.kernel == "gaussian":
            x = self.gaussian_kernel(x, self.initial)
        predictions = [np.dot(x, self.weights[i]) + self.biases[i] for i in range(self.n_classes)]
        return np.argmax(predictions, axis=0)



    
    
    
    