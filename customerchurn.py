# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:39:12 2020

@author: DELL
"""

import numpy as np
import pandas as pd 
from google.colab import files 
file = files.upload()
ex = pd.read_csv("/content/BankCustomers.csv",header=None)
ex.head(10)

X = ex.iloc[1:,3:11]
y = ex.iloc[1:,11:]
print(X.shape) 
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(y_test)
print(X_train.shape)
print(y_train.shape)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape)

import keras 
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=8, units=5, kernel_initializer="uniform"))
classifier.add(Dense(activation="relu", units=5, kernel_initializer="uniform"))
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train,y_train, batch_size = 10, nb_epoch = 50)


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

accuracy = confusion_matrix(y_test, y_pred)
