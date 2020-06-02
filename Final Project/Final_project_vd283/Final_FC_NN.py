import math
import numpy as np  
from download_mnist import load
import operator  
import time
from time import time

#hyperparameters
step_size=0.01
reg=1e-3
num=60000
epslon=0.0001
b=0.8 #momentum
 
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

x_train1 =x_train[:,range(num)]
y_train1=y_train[:,range(num)]



#Random Initialisation of W arrays for  784‐200‐50‐10
w_1=np.random.randn(200,784)*0.01
b_1=np.zeros((200,1))

w_2=np.random.randn(50,200)*0.01
b_2=np.zeros((50,1))


w_3=np.random.randn(10,50)*0.01
b_3=np.zeros((10,1))


batch_size=128

v_w1=np.zeros((200,784))
v_w2=np.zeros((50,200))
v_w3=np.zeros((10,50))

v_b1=np.zeros((200,1))
v_b2=np.zeros((50,1))
v_b3=np.zeros((10,1))

t0 = time()
for i in range(10): # Number of Epochs
    
    batch_start=0
    y=batch_size
    x=num
    total_loss=0
    
    

    while(x > 0):
        
        x_train=x_train1[:,range(batch_start,batch_start+y)]
        y_train=y_train1[:,range(batch_start,batch_start+y)]

        #Forward Propagation

        #Layer 1 linear regression
        
        z_1=np.dot(w_1,x_train)+b_1  #here b_1 is broadcasted for the 60000 training images
        a_1=np.maximum(0,z_1) # Relu activation function
        
        #Layer 2 linear regression

        z_2=np.dot(w_2,a_1)+b_2  #here b_2 is broadcasted for the 60000 training images
        a_2=np.maximum(0,z_2)
        
            
        #Layer 3 Logistic regression for output
        z_3=np.dot(w_3,a_2)+b_3  #here b_3 is broadcasted for the 60000 training images
        
        
        exp_scores=np.exp(z_3)
        a_3=exp_scores/np.sum(exp_scores,axis=0,keepdims=True)
        
        #Loss Function
        loss=-np.log(a_3[y_train,range(0,y)])
        f_loss=np.sum(loss)/y
                        
        #need to add regularisation loss
        reg_loss=0.5 * reg * (np.sum(np.square(w_3))+np.sum(np.square(w_2))+np.sum(np.square(w_1)))

        total_loss+= (f_loss + reg_loss)
        
        #Backward Propagation
        dz_3=a_3
        dz_3[y_train,range(0,y)]-=1


        dw_3=np.dot(dz_3,np.transpose(a_2))
        dw_3/=y
        db_3=np.sum(dz_3,axis=1,keepdims=True)
        db_3/=y

        # back propagation into layer 2
        dz_2=np.dot(np.transpose(w_3),dz_3)
        dz_2[z_2 <= 0]=0 # Multiplied by diffrentiation of activation function
        dw_2=np.dot(dz_2,np.transpose(a_1))
        dw_2/=y
        db_2=np.sum(dz_2,axis=1,keepdims=True)
        db_2/=y

        # back propagation into layer 1
        dz_1=np.dot(np.transpose(w_2),dz_2)
        dz_1[z_1 <= 0]=0 # Multiplied by diffrentiation of activation function
        dw_1=np.dot(dz_1,np.transpose(x_train))
        dw_1/=y
        db_1=np.sum(dz_1,axis=1,keepdims=True)
        db_1/=y
        
        #adding regularisation loss
        dw_3+=reg*w_3
        dw_2+=reg*w_2
        dw_1+=reg*w_1
     
        #Calculate the moving average
        v_w1= (1-b)*dw_1 + b*v_w1
        v_w2= (1-b)*dw_2 + b*v_w2
        v_w3= (1-b)*dw_3 + b*v_w3
        
        v_b1= (1-b)*db_1 + b*v_b1
        v_b2= (1-b)*db_2 + b*v_b2
        v_b3= (1-b)*db_3 + b*v_b3

        #Learning based on the derivatives
        '''w_1+= (-step_size * dw_1)   
        w_2+= (-step_size * dw_2)
        w_3+= (-step_size * dw_3)
        b_1+= (-step_size * db_1)
        b_2+= (-step_size * db_2)
        b_3+= (-step_size * db_3)'''

        #Learning based on the moving averages
        w_1+= (-step_size * v_w1)   
        w_2+= (-step_size * v_w2)
        w_3+= (-step_size * v_w3)
        b_1+= (-step_size * v_b1)
        b_2+= (-step_size * v_b2)
        b_3+= (-step_size * v_b3)


        x-=batch_size
        batch_start+=batch_size
        if(x < batch_size-1):
            y=x
   
    #printing of the total loss per 1000 increment in number of iterations
    print("The Total average loss after the epoch number ",i,"is : ",total_loss/batch_size)
t1 = time() 

# Testing Phase
exp_=np.exp(np.dot(w_3,np.maximum(0,np.dot(w_2,np.maximum(0,np.dot(w_1,x_test)+b_1))+b_2))+b_3)
result=np.argmax((exp_/np.sum(exp_,axis=0,keepdims=True)), axis=0)
print("Training accuracy : ", np.mean(result==y_test))
print("Training Time taken: ",(t1-t0))