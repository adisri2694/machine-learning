

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_training_data(path):  
    data = pd.read_csv(path)

   
    data.insert(0, 'Ones', 1)

    num_columns = data.shape[1]                       
    panda_X = data.iloc[:,0:num_columns-1]            
    panda_y = data.iloc[:,num_columns-1:num_columns]  

    X = np.matrix(panda_X.values)   
    y = np.matrix(panda_y.values)   

    return X, y

def compute_mean_square_error(X, y, theta):
    summands = np.power(X * theta.T - y, 2)
    return np.sum(summands) / (2 * len(X))

def gradient_descent(X, y, learning_rate, num_iterations):
    num_parameters = X.shape[1]                                 
    theta = np.matrix([0.0 for i in range(num_parameters)])     #
    cost = [0.0 for i in range(num_iterations)]

    for it in range(num_iterations):
        error = np.repeat((X * theta.T) - y, num_parameters, axis=1)
        error_derivative = np.sum(np.multiply(error, X), axis=0)
        theta = theta - (learning_rate / len(y)) * error_derivative
        cost[it] = compute_mean_square_error(X, y, theta)

    return theta, cost

X, y = get_training_data('d1.txt')
theta, cost = gradient_descent(X, y, 0.008, 10000)

print('Theta: ', theta)
print('Cost: ', cost[-1])
plt.plot(cost,np.arange(10000))
