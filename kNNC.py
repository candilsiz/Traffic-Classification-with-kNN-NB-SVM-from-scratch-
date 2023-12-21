#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:02:37 2023

@author: candilsiz
"""

import numpy as np
from collections import Counter
    
class KNNClassifier:
    
    def __init__(self, k):
        self.k = k
          
    def fit(self, X, y):        
        self.featureTrain = X
        self.labelTrain = y       
      
    def euclidean_distance(self, dataPoint1, dataPoint2):
        return np.sqrt(np.sum((dataPoint1 - dataPoint2) ** 2))
    
    def predict(self, dataPoints):
        labelPrediction = [self.predict_single(singleData) for singleData in dataPoints]
        return np.array(labelPrediction)
    
    def predict_single(self, x): 
        distances = [self.euclidean_distance(x, feature_train) for feature_train in self.featureTrain]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.labelTrain.iloc[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common
       
    # def cosine_distance(self, dataPoint1, dataPoint2):
    #     dot_product = np.dot(dataPoint1, dataPoint2)
    #     norm_dataPoint1 = np.linalg.norm(dataPoint1)
    #     norm_dataPoint2 = np.linalg.norm(dataPoint2)
    #     return 1 - dot_product / (norm_dataPoint1 * norm_dataPoint2)


