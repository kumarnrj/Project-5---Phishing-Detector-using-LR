# -*- coding: utf-8 -*-
"""
Created on Wed May 29 20:50:16 2019

@author: NR
"""
#importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset using pandas
dataset = pd.read_csv('phishing.txt')

# visualization 
dataset.describe()
#separating the features and label
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,30].values

#Creating the traing set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3 , random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=100,random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#score 
classifier.score(X_train,y_train)
classifier.score(X_test,y_test)


#*************************************************************
#Train with only two input parameters - parameter Prefix_Suffix and 13 URL_of_Anchor.

X1 = dataset.iloc[:,[5,13] ].values
y1 = dataset.iloc[:,30].values

#Creating the traing set 
from sklearn.cross_validation import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size =0.3 , random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=100,random_state = 0)
classifier.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred1)

#score 
classifier.score(X_train1,y_train1)
classifier.score(X_test1,y_test1)





# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#score 
classifier.score(X_train,y_train)
classifier.score(X_test,y_test)

