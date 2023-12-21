#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:55:59 2023

@author: candilsiz
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class PCA:
    
    def pca(self, x, num_components):
        
        """ 
        
            1 - Data Standardization
            2 - Find Covariance Matrix of Standardized Data
            3 - Compute Eigen Vectors and Eigen Values of Covariance Matrix
            4 - Sort Eigen Values in Descending Order
            5 - Original Data is projected onto New Feature Space (Selected Eigen Vectors forms the Matrix)
            
        """
        
        x_Standard = StandardScaler().fit_transform(x)
        covariance_matrix = np.cov(x_Standard.T)
        eigenValues, eigenVectors = np.linalg.eig(covariance_matrix)
        
        top_eigen_vecs = eigenVectors[:, :num_components]
        xPCA = x_Standard.dot(top_eigen_vecs)
        
        # PCA finishes, Scree Plot starts
        sorted_eigen_vals = sorted(eigenValues, reverse=True)
        totalVariance = sum(sorted_eigen_vals)
        varianceRatio = [values / totalVariance for values in sorted_eigen_vals]
        
        plt.figure(figsize=(8, 5))
        plt.plot(varianceRatio[:num_components], marker='o')
        plt.xlabel('Principal Components')
        plt.ylabel('Variance Ratio')
        plt.title('Principal Components vs Variance Ratio')
        plt.show()
        
        return xPCA






