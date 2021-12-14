# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 22:15:54 2021

@author: Pierre.M
"""


//----

import pandas as pd # For data manipulation and analaysis.
import numpy as np # For data multidimentional collections and mathematical operations.
# For statistics Plotting Purpose
import matplotlib.pyplot as plt

# For Classification Purpose
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
    


dataset = pd.read_csv('wine.csv')

# Preprocessing Phase

# Checking having missing values
print(dataset['color'].value_counts())

# Replace missing values (NaN) with bulbbous stalk roots
dataset['high_quality'].replace(np.nan, '1', inplace = True)

# Encoding textual values: Converting lingustic values to numerical values
mappings = list()
encoder = LabelEncoder()
for column in range(len(dataset.columns)):
    dataset[dataset.columns[column]] = encoder.fit_transform(dataset[dataset.columns[column]])
    mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
    mappings.append(mappings_dict)
    
# Separating class color from the dataset features 
X = dataset.drop('color', axis=1)
y = dataset['color']

# Splitting dataset to training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle =True, test_size=0.3, random_state=42)
DTC = DecisionTreeClassifier()
DTC.fit(X_train,y_train)
predDTC = DTC.predict(X_test)
reportDTC = classification_report(y_test,predDTC, output_dict = True)
crDTC = pd.DataFrame(reportDTC).transpose()
print(crDTC)

# Tree Visualisation

fig = plt.figure(figsize=(100,80))
plot = plot_tree(DTC, feature_names=list(dataset.columns), class_names=['RED', 'White'],filled=True)
for i in plot:
    arrow = i.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('black')
        arrow.set_linewidth(2)
