import numpy as np
import pandas as pd
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values
#x.reshape(-1,1)
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
re=LinearRegression()
re.fit(x_train.reshape(24,1), y_train.reshape(24,1))
y_p=re.predict(x_test.reshape(6,1))

print reg.score(x.reshape(30,1),y.reshape(30,1))