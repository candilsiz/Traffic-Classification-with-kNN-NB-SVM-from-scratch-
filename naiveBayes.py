#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:09:54 2023

@author: candilsiz
"""

import numpy as np
from sklearn.base import BaseEstimator

class NaiveBayesClassifier(BaseEstimator):
    
    def fit(self, X, y):
        
        num_samples, num_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        # Initilize mean, variance, and priors
        self._mean = np.zeros((n_classes, num_features), dtype=np.float64)
        self._var = np.zeros((n_classes, num_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        
        # For each class label calculate mean, variance, and prior
        for idx, classy in enumerate(self._classes):
            XClass = X[y == classy]
            self._mean[idx, :] = XClass.mean(axis=0)
            self._var[idx, :] = XClass.var(axis=0)
            self._priors[idx] = XClass.shape[0] / float(num_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        for idx, i in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            eps = 1e-6
            classConditional = np.sum(np.log(self._pdf(idx, x) + eps))
            posterior = prior + classConditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]
    
    # Calculate PDF of Gaussian Distribution 
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        variance = self._var[class_idx]
        variance += 1e-4
        # Apply PDF of Gaussian Distribution 
        numerator = np.exp(- (x - mean) ** 2 / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator
    
    