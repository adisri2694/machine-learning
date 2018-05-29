#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 09:38:28 2018

@author: aditya
"""

import numpy as np
import pandas as pd
dataset=pd.read_csv('50-Startups.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:, 3]=labelencoder_x.fit_transform(x[:, 3])
onehotencoder= OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

print regressor.score(x,y)