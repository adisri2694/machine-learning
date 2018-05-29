import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path='d1.txt'
data=pd.read_csv(path,header=None, names=['population','profit'])
data.head()

#data.plot(kind='scatter',x='population', y='profit',figsize=(12,8))
x=data.iloc[:,0]
y=data.iloc[:,1]

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x.reshape(97,1),y.reshape(97,1))
yp=reg.predict(x.reshape(97,1))
data.plot(kind="scatter",x='population',y='profit',figsize=(12,8))
plt.plot(x,yp)
reg.score(x.reshape(97,1),y.reshape(97,1))





