import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('titanic_train.csv')
x=dataset.iloc[:,1:]  #deleting passenger id column as it has no realtion with the prediction
y=x.iloc[:,0].reshape(891,1)

x.insert(0, 'Ones', 1) #inserting a column of one with respect to theta zero

x=x.drop(['Survived','Name','Ticket','Cabin'],axis=1) #droping name as it has no effect, ticket as it 
                                                      #is very ambiguous data and cabin because it is mostly NAN

for i in range(0,8):                          #number of NAN in each rows
  NoOfNan=len(x.iloc[:,i])-x.iloc[:,i].count()
  print NoOfNan
#
x.iloc[:,3].fillna(x.iloc[:,3].mean(),inplace=True)  #filling NAN values with means

from sklearn.preprocessing import LabelEncoder      #importing LAbelEncoder class
lblenc=LabelEncoder()                               #creating its object
x.iloc[:,2]=lblenc.fit_transform(dataset.iloc[:,3]) #encoding columns which are not numbers
x.iloc[:,7]=lblenc.fit_transform(dataset.iloc[:,7])

from sklearn import preprocessing            #normalising
x = preprocessing.normalize(x)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


theta=np.zeros([1,x_train.shape[1]])       #defining theta and learning rate similarly as linear regression 
alpha=0.001
iterations=10000

def logistic(x,theta):                    #defining a sigmoid function which returns a value between 0 and 1
    a=np.dot(x,theta.T)
    h=1/(1+np.exp(-a))
    return h



def cost(x, y, theta):                 #defining a cost function  which uses log function and sigmoid values and is convex
    h = logistic(x, theta)             #the squared mean error funtion is not convex in logical regression case
    cost = np.sum(np.dot(-y.T, np.log(h)) - np.dot((1 -y.T), np.log(1 - h)))/len(x)
    return cost


def gradientDescent(x , y,theta,alpha,iters):
    costarray=np.zeros(iters)  #.reshape(iters,1)
    for i in range(iters):
       h = logistic(x, theta)
       gradient =np.sum( np.dot(x.T, (h - y)),axis=1) ; #calculating the gradient  
       theta = theta - alpha * gradient          #updating alpha
       
       costarray[i]=cost(x,y,theta)
       
    return theta,costarray


g,cost1 = gradientDescent(x_train,y_train,theta,alpha,iterations)
print("final theta")
print(g)

finalCost = cost(x_train,y_train,g)
print("final cost"+str(finalCost))

plt.plot(np.arange(iterations),cost1,'*')



y_pd=logistic(x_test,g)

for i in range(x_test.shape[0]):
    if(y_pd[i]>=0.5):              #defining a decision boundary for the algorithm
        y_pd[i]=1
    else:
        y_pd[i]=0
        


count=0;

for i in range (y_pd.shape[0]):   #calculating accuracy using the number of right prdictions and total number of predictions
    if(y_pd[i]==y_test[i]):
        count=count+1
    
accuracy=((count)*100/(y_pd.shape[0]))

print str(accuracy)+"%"
    
    
        
    




 

