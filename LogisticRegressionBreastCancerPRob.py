import numpy as np
import pandas as pd
dataset=pd.read_csv('breastCancerdata.csv').iloc[:,:-1]
x=dataset.iloc[:,2:].values
y=dataset.iloc[:,1].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
'''onehotencoder=OneHotEncoder(y)
y=onehotencoder.fit_transform(y).toarray()'''
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)
y_p=reg.predict(x_test)
print reg.score(x,y)

