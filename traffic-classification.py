#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:55:59 2023

@author: candilsiz
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from kNNC import KNNClassifier
from naiveBayes import NaiveBayesClassifier
from svm import SupportVectorMachineClassifier
from pca import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


weekend_days = ['Saturday', 'Sunday'] 
labels = [3,1,2,4]
traffic_mapping = {"low": 0, "normal": 1, "high": 2, "heavy": 3}
traffic = pd.read_csv('Traffic.csv')
traffic['Time'] = pd.to_datetime(traffic['Time'], format='%I:%M:%S %p')
traffic['Hour'] = traffic['Time'].dt.hour
traffic['Day Type'] = traffic['Day of the week'].apply(lambda x: 1 if x in weekend_days else 0)
traffic['Time of Day'] = pd.cut(traffic['Hour'], bins=[0, 6, 12, 18, 24], labels = labels, right = False, include_lowest = True)
traffic['Commutting Hour'] = traffic['Hour'].apply(lambda x: 1 if 7 <= x <= 9 or 17 <= x <= 19 else 0)
traffic["Traffic Situation"] = np.array([traffic_mapping[label] for label in traffic["Traffic Situation"]])

# Low correlations ["Date", "Day of the Week", "Day Type"]
corr = traffic.corr(method='pearson')
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

# Correlation matrix inference
traffic = traffic.drop(columns=["Date", "Hour", "Time", "Time of Day","Day of the week",], axis=1)

corr = traffic.corr(method='pearson',numeric_only=True)
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

label = traffic["Traffic Situation"]
features = traffic.drop(columns = ["Traffic Situation"])

preprocessor = ColumnTransformer(
    transformers=[
        #('num', StandardScaler(), ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total'])
        ('num', StandardScaler(), ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total','Commutting Hour','Day Type'])
        #('cat', OneHotEncoder(), ['Day Type', ,'Commutting Hour'])
    ])

featureTrain, featureTest, labaelTrain, labelTest = train_test_split(features, label, test_size=0.15, random_state=42)

featureTrain = preprocessor.fit_transform(featureTrain)
featureTest = preprocessor.transform(featureTest)        

# pca_object = PCA() # Decreases accuracy and other metrics
# featureTrain = pca_object.pca(featureTrain, num_components=3)
# featureTest = pca_object.pca(featureTest, num_components=3)

knn = KNNClassifier(k=4)
nb = NaiveBayesClassifier()
svm = SupportVectorMachineClassifier(C=10, features=7, sigma_sq=0.04, kernel="gaussian", n_classes=4)

knn.fit(featureTrain, labaelTrain)
nb.fit(featureTrain, labaelTrain)
svm.fit(featureTrain, labaelTrain, epochs=1, learning_rate=0.05)

labelPred_knn = knn.predict(featureTest)
labelPred_nb = nb.predict(featureTest)
labelPred_svm = svm.predict(featureTest)

accuracy_knn = accuracy_score(labelTest, labelPred_knn)
precision_knn = precision_score(labelTest, labelPred_knn, average='weighted')
recall_knn = recall_score(labelTest, labelPred_knn, average='weighted')
f1_knn = f1_score(labelTest, labelPred_knn, average='weighted')

accuracy_nb = accuracy_score(labelTest, labelPred_nb)
precision_nb = precision_score(labelTest, labelPred_nb, average='weighted')
recall_nb = recall_score(labelTest, labelPred_nb, average='weighted')
f1_nb = f1_score(labelTest, labelPred_nb, average='weighted')

accuracy_svm = accuracy_score(labelTest, labelPred_svm)
precision_svm = precision_score(labelTest, labelPred_svm, average='weighted')
recall_svm = recall_score(labelTest, labelPred_svm, average='weighted')
f1_svm = f1_score(labelTest, labelPred_svm, average='weighted')

print("\nKNN Classifier Performance:")
print("Accuracy:", accuracy_knn)
print("Precision:", precision_knn)
print("Recall:", recall_knn)
print("F1 Score:", f1_knn)

print("\nNaive Bayes Classifier Performance:")
print("Accuracy:", accuracy_nb)
print("Precision:", precision_nb)
print("Recall:", recall_nb)
print("F1 Score:", f1_nb)

print("\nSupport Vector Machine Classifier Performance:")
print("Accuracy:", accuracy_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1 Score:", f1_svm)


model_preds = [labelPred_knn, labelPred_nb, labelPred_svm]
model_names = ["k-Nearest Neighbour", "Naive Bayes", "Support Vector Machine"]

i=0

for prediction in model_preds:
    
    conf_matrix = confusion_matrix(labelTest, prediction)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(4), yticklabels=range(4)) 
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix of {model_names[i]}')
    plt.show()
    i+=1







