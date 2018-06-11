import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing



from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
x_train=mnist.train.images.reshape(55000,28,28)
y_train=mnist.train.labels
numberdata=150
x_t=x_train[0:numberdata,:,:]
x_test=mnist.test.images.reshape(10000,28,28)
y_test=mnist.test.labels

x_test_b=x_test[0:100,:,:]
y_test_b=y_test[0:100]
 

for i in range (x_t.shape[1]):
   for j in range (x_t.shape[2]):
       mean=np.mean(x_t[:,i,j])
       std=np.std(x_t[:,i,j])
       for e in range (x_t.shape[0]):
            x_t[e,i,j]=(x_t[e,i,j]-mean)/std
            if (np.isnan(x_t[e,i,j])):
                x_t[e,i,j]=0

y_t=y_train[0:numberdata]
l1,l2,l3=0.0001,0.0001,0.0001
#plt.gray() # use this line if you don't want to see it in color
#plt.imshow(x_train[2].reshape(28,28))
#plt.show()
              
#Z is activated matrix after 1st convolution                
     
def relu(image):
    #print"inside relu image size",image.shape
    image[image<0]=0
    return image

def zero_pad(X, pad):
     X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values = 0)
    
     return X_pad
 
    
def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev,W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + float(b)
    return Z

def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    
    (f, f, n_C_prev, n_C) = W.shape
    
    
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    n_H = int(np.floor((n_H_prev-f+2*pad)/stride)) + 1
    n_W = int(np.floor((n_W_prev-f+2*pad)/stride)) + 1
    
    Z = np.zeros((m,n_H,n_W,n_C))
    
    A_prev_pad = zero_pad(A_prev,pad)
    
    for i in range(m):                               
        a_prev_pad = A_prev_pad[i]                              
        for h in range(n_H):                          
            for w in range(n_W):                       
                for c in range(n_C):                   
                    
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    
                    Z[i, h, w, c] = conv_single_step(a_slice_prev,W[:,:,:,c],b[:,:,:,c])
                                        
    
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache

def pool_forward(A_prev, hparameters, mode = "max"):
   
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):                 
                for c in range (n_C):
                    
                    vert_start = (h)*stride
                    vert_end = vert_start+f
                    horiz_start = (w)*stride
                    horiz_end = horiz_start+f
                    
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    
    cache = (A_prev, hparameters)
    
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache




#def softmax(z2):
#   z2 = np.exp(z2)
#   sums =  np.sum(z2,axis = 1)
#   sums = sums.reshape((numberdata,1))
#   A2 = z2/sums
    
   #return A2
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_classifier(softmax_output):
    predicted=np.zeros(shape=(softmax_output.shape[0],1))
    predicted=np.argmax(softmax_output,axis=1)
    return predicted



def conv_backward(dZ, cache):
   
    
    (A_prev, W, b, hparameters) = cache
    
   
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
   
    (f, f, n_C_prev, n_C) = W.shape
    
    
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros_like(A_prev)                           
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev,pad)
    dA_prev_pad = zero_pad(dA_prev,pad)
    
    for i in range(m):                       # loop over the training examples
        
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h*stride
                    vert_end = vert_start+f
                    horiz_start = w*stride
                    horiz_end = horiz_start+f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i,h,w,c]
                    dW[:,:,:,c] += a_slice*dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]
                    
        
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db


def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask

def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz
    a = average*np.ones((n_H,n_W))/(n_H*n_W)
    return a








def pool_backward(dA, cache, mode = "max"):
   
    (A_prev, hparameters_pool) = cache
    
    
    stride = hparameters_pool['stride']
    f = hparameters_pool['f']
    
   
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.reshape(numberdata,26,26,2).shape
    
    
    dA_prev = np.zeros_like(A_prev)
    
    for i in range(m):                       # loop over the training examples
        
       
        a_prev = A_prev[i]
        
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    
                   
                    vert_start = h *stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f
                    
                  
                    if mode == "max":
                        
                        
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        
                        mask = create_mask_from_window(a_prev_slice)
                       
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]
                        
                    elif mode == "average":
                        
                      
                        da = dA[i,h,w,c]
                       
                        shape = (f,f)
                        
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da,shape)
                        
    
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev
def batchnorm_forward(x, gamma, beta, eps):

  N, D = x.shape

  #step1: calculate mean
  mu = 1./N * np.sum(x, axis = 0)

  #step2: subtract mean vector of every trainings example
  xmu = x - mu

  #step3: following the lower branch - calculation denominator
  sq = xmu ** 2

  #step4: calculate variance
  var = 1./N * np.sum(sq, axis = 0)

  #step5: add eps for numerical stability, then sqrt
  sqrtvar = np.sqrt(var + eps)

  #step6: invert sqrtwar
  ivar = 1./sqrtvar

  #step7: execute normalization
  xhat = xmu * ivar

  #step8: Nor the two transformation steps
  gammax = gamma * xhat

  #step9
  out = gammax + beta

  #store intermediate
  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache

def batchnorm_backward(dout, cache):

  #unfold the variables stored in cache
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache

  #get the dimensions of the input/output
  N,D = dout.shape

  #step9
  dbeta = np.sum(dout, axis=0)
  dgammax = dout #not necessary, but more understandable

  #step8
  dgamma = np.sum(dgammax*xhat, axis=0)
  dxhat = dgammax * gamma

  #step7
  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar

  #step6
  dsqrtvar = -1. /(sqrtvar**2) * divar

  #step5
  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

  #step4
  dsq = 1. /N * np.ones((N,D)) * dvar

  #step3
  dxmu2 = 2 * xmu * dsq

  #step2
  dx1 = (dxmu1 + dxmu2)
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

  #step1
  dx2 = 1. /N * np.ones((N,D)) * dmu

  #step0
  dx = dx1 + dx2

  return dx, dgamma, dbeta

gamma1=1
beta1=0
eps=0.0001
np.random.seed(3)
W=np.random.randn(3,3,1,2)

np.random.seed(3)
b=np.random.randn(1,1,1,2)

hparameters={"stride":1,"pad":1,"f":3}
x_tt=x_t.reshape(numberdata,28,28,1)
#z=np.zeros(50000,28,28,2)
product=26*26*2

np.random.seed(3)
W1=np.random.randn(15,product)

B1=np.zeros((15,1))

np.random.seed(3)
W2 = np.random.randn(10,15)

b2 = np.zeros((10,1))
loss1=[]
i=0
for i in range (1,5):
  
  z , cachec = conv_forward(x_tt,W,b,hparameters)
  z=relu(z)
  
  #print z.shape
  
  z1, cachep =pool_forward(z,hparameters,"max")
  
 
  
  product=z1.shape[1]*z1.shape[2]*z1.shape[3]
  
  final_input=z1.reshape(z1.shape[0],product)
  batch_out, cache_batch = batchnorm_forward(final_input,gamma1,beta1,eps) 
  
  Z1 = np.dot(batch_out,W1.T) + B1.T
  A1 = 1/(1+np.exp(-Z1))

  Z2 = np.dot(A1,W2.T) + b2.T

  Z3=softmax(Z2)
  #print Z3.shape
  
  Z4=softmax_classifier(Z3)
 #print Z4
  
  Y = np.eye(numberdata,10)[y_t.reshape(-1)]
  loss =  -sum(np.sum(Y*np.log(Z3),axis = 1))/len(Y)
  loss1.append(loss)
  
  dZ2 = Z3 - Y

  dW2 = np.dot(dZ2.T , A1)
  db2 = np.sum(dZ2,axis= 0,keepdims = True)

  W2 = W2 - l3 * dW2
  b2 = b2 - l3 * db2.T

  dZ1 = np.dot(dZ2,W2) * ((1-A1)*A1)
  dW1 = np.dot(dZ1.T , final_input)
  db1 = np.sum(dZ1,axis= 0,keepdims = True)

  W1 = W1 - l2 * dW1
  B1 = B1 - l2 * db1.T

  dA0 = np.dot(dZ1,W1)
  
  dA_b, dgamma1, dbeta1 = batchnorm_backward(dA0,cache_batch)
  gamma1 = gamma1 - 0.01 * dgamma1
  beta1 = beta1 - 0.01 * dbeta1
  



  dA_b= dA_b.reshape((numberdata,26,26,2))

  dA_conv = pool_backward(dA0.reshape(numberdata,26,26,2) ,cachep,mode = 'max')

  dA_prev, dW, db = conv_backward(dA_conv,cachec)

  W = W - l1 * dW
  b = b - l1 * db

plt.plot(np.arange(i),loss1)

zt1,ca=conv_forward(x_test_b.reshape(100,28,28,1),W,b,hparameters)
zt1=relu(zt1)
zt2,ca1=pool_forward(zt1,hparameters,"max")

 
  
product=zt2.shape[1]*zt2.shape[2]*zt2.shape[3]
  
final_input=zt2.reshape(zt2.shape[0],product)
  
Zt2 = np.dot(final_input,W1.T) + B1.T
A1 = 1/(1+np.exp(-Zt2))

Zt3 = np.dot(A1,W2.T) + b2.T

Zt4=softmax(Zt3)
Zt5=softmax_classifier(Zt4)

c=0
for i in range(Zt5.shape[0]):
   if(Zt5[i]==y_test_b[i]):
       c=c+1
accuracy=c
print("accuracy",str(c)+"%")

