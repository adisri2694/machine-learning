import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('winequality-red.csv',delimiter=';')
dataset.insert(0, 'Ones', 1)

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1].reshape(1599,1) #reshaping rank-1 arrays to prevent any errors

from sklearn import preprocessing  #normalizing the data to increse the efficiency of gradient descent
x = preprocessing.normalize(x)     #data having scaled features and closeby values tend to descent better and quicker
print np.isnan(np.min(x))          #this function prints the minimum value in the dataset,if this value is not NAN 
                                   #this means that there is no NAN values in the dataset

from sklearn.cross_validation import train_test_split #splitting the data into test and train dataframes
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



theta=np.zeros([1,x.shape[1]])  #initializing theta as a row vector of zero,though the value with which it is initialised 
                                #doesn't matters 
alpha=0.001                     #learning rate ,the bigger it is the faster convergence, but there is a chance of overshooting
                                #in case of high learning rate values 
iterations=10000

def computeCost(x,y,theta):    #function to calculate cost a hypothesis  
    a = np.power((np.dot(x , theta.T)-y),2)
    return np.sum(a)/(2 * len(x))

def gradientDescent(x,y,theta,iters,alpha): #to perform gradient descent
    cost = np.zeros(iters)                  #similar case as initializing of theta
    cost= cost.reshape(iterations,1)        
    for i in range(iters):                  #performing iterations
        theta = theta - (alpha/len(x)) * np.sum(np.dot(x.T , (np.dot(x , theta.T) - y)), axis=1) 
                                                             #updating theta using the gradient of the cost function 
                                                             #this is an example of batch graident descent as whole of the dataset is used
                                                             #to perform the operation
                                                                                                   
        cost[i] = computeCost(x, y, theta)
    
    return theta,cost      #returns the final parameters and cost array

g,cost = gradientDescent(x_train,y_train,theta,iterations,alpha)
print("the parameters are :")
print g 

finalCost = computeCost(x_train,y_train,g)
print("final cost:"+str(finalCost))

plt.plot(np.arange(10000),cost) #plot of cost with number of iterations,

y_pd=(np.dot(g,x_test.T)).T   #testing the data by putiing the test data into our hypothesis theta.T*x

mean=(np.sum(y_test))/(y_test.shape[0])

 
SSres=np.sum(np.power((y_test-y_pd),2))     #calculating score by measuring R square known as coefficient of determination
SStot=np.sum(np.power((y_test-mean),2))

score=1-(SSres/SStot)
print ("score: "+str(score))

