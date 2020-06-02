import math
import numpy as np  
from download_mnist import load
import operator  
from time import time

#hyperparameters
step_size=0.01
reg=1e-3
num=60000
epslon=0.0001

# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,784)
x_test  = x_test.reshape(10000,784)
y_train=y_train.reshape(1,60000)
y_test=y_test.reshape(1,10000)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
y_train=y_train.astype(int)


#Transposing the test 
x_train = x_train.transpose()
x_test=x_test.transpose()

x_train =x_train[:,range(num)]
y_train=y_train[:,range(num)]

#Normalize the Training set
#x_train_mean=np.sum(x_train , axis=1 , keepdims=True)/num
#x_train=x_train-x_train_mean # Broadcasting is used

#x_train_variance=np.sum(np.square(x_train), axis=1 , keepdims=True)/num
#x_train=x_train/(np.sqrt(x_train_variance+ epslon))


#Random Initialisation of W arrays for  784‐200‐50‐10
w_1=np.random.randn(200,784)*0.01
b_1=np.zeros((200,1))

w_2=np.random.randn(50,200)*0.01
b_2=np.zeros((50,1))

w_3=np.random.randn(10,50)*0.01
b_3=np.zeros((10,1))

t0 = time()
for i in range(500):

    #Forward Propagation

    #Layer 1 linear regression
    
    z_1=np.dot(w_1,x_train)+b_1  #here b_1 is broadcasted for the 60000 training images
    a_1=np.maximum(0,z_1) # Relu activation function
    
    #Layer 2 linear regression

    z_2=np.dot(w_2,a_1)+b_2  #here b_2 is broadcasted for the 60000 training images
    a_2=np.maximum(0,z_2)
    
    
    
    #Layer 3 linear regression for output

    z_3=np.dot(w_3,a_2)+b_3  #here b_3 is broadcasted for the 60000 training images
    
    
    exp_scores=np.exp(z_3)
    a_3=exp_scores/np.sum(exp_scores,axis=0,keepdims=True)
    
    #Loss Function
    loss=-np.log(a_3[y_train,range(num)])
    f_loss=np.sum(loss)/num
                       
    #need to add regularisation loss
    reg_loss=0.5 * reg * (np.sum(np.square(w_3))+np.sum(np.square(w_2))+np.sum(np.square(w_1)))

    total_loss= f_loss + reg_loss
    
    #printing of the total loss per 1000 increment in number of iterations
    print("The Total loss in these ",i," iterations is : ",total_loss)

    #Backward Propagation
    dz_3=a_3
    dz_3[y_train,range(num)]-=1


    dw_3=np.dot(dz_3,np.transpose(a_2))
    dw_3/=num
    db_3=np.sum(dz_3,axis=1,keepdims=True)
    db_3/=num

    # back propagation into layer 2
    dz_2=np.dot(np.transpose(w_3),dz_3)
    dz_2[z_2 <= 0]=0 # Multiplied by diffrentiation of activation function
    dw_2=np.dot(dz_2,np.transpose(a_1))
    dw_2/=num
    db_2=np.sum(dz_2,axis=1,keepdims=True)
    db_2/=num

    # back propagation into layer 1
    dz_1=np.dot(np.transpose(w_2),dz_2)
    dz_1[z_1 <= 0]=0 # Multiplied by diffrentiation of activation function
    dw_1=np.dot(dz_1,np.transpose(x_train))
    dw_1/=num
    db_1=np.sum(dz_1,axis=1,keepdims=True)
    db_1/=num
    
    #adding regularisation loss
    dw_3+=reg*w_3
    dw_2+=reg*w_2
    dw_1+=reg*w_1

    #Learning based on the derivatives
    w_1+= (-step_size * dw_1)
    
    
    w_2+= (-step_size * dw_2)
    w_3+= (-step_size * dw_3)
    b_1+= (-step_size * db_1)
    b_2+= (-step_size * db_2)
    b_3+= (-step_size * db_3)

t1 = time()

# Testing Phase
exp_=np.exp(np.dot(w_3,np.maximum(0,np.dot(w_2,np.maximum(0,np.dot(w_1,x_test)+b_1))+b_2))+b_3)
result=np.argmax((exp_/np.sum(exp_,axis=0,keepdims=True)), axis=0)

t2 = time()

print("Training accuracy : ", np.mean(result==y_test))
print("Training Time taken: ",(t1-t0))
print("Testing Time taken: ",(t2-t1))