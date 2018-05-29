import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Advertising.csv")
dataset.drop(dataset.columns[0], axis=1,inplace=True)  # df.columns is zero-based pd.Index 

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,3]

#dataset.plot(kind='scatter',x='radio',y='sales',figsize=(12,8))
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_p=regressor.predict(x_test)

#dataset.plot(kind="scatter",x='TV',y='sales',figsize=(12,8))
plt.plot(x_test.iloc[:,2],y_p,"ro")
print regressor.score(x,y)