import math
import numpy as np  
from download_mnist import load
import operator  
import time
import torch
import torch.nn as nn
import torch.nn.functional as F    
import torch.optim as optim
from torchvision import datasets,transforms
from time import time


#Dataset
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)


# Neural Network for  784‐200‐50‐10
model=nn.Sequential(nn.Linear(784,200),nn.ReLU(),nn.Linear(200,50),nn.ReLU(),nn.Linear(50,10),nn.Softmax())

def train(model , train_loader , optimiser , loss, epoch):
    
    #started for training
    model.train()
    count=0
    for batch_id,(data,target) in enumerate(train_loader):
        #data,target=data.to(device),target.to(device)
        count+=1
        optimiser.zero_grad()#resetting the gradient
        
        data=data.view(-1,784)#------
        
        output=model(data)#data is passed to the model set so as to come out with output of forward propagation
        
        #Conversion of labels in the target as float data type
        
        loss_fn=nn.CrossEntropyLoss()#Using cross entropy loss
        loss=loss_fn(output,target)#calculate the loss based on the output and the target
        loss.backward()
        optimiser.step()
    

def test(model,test_loader):
    count=0
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad() :
        for data,target in test_loader:
            count+=1
            #data,target=data.to(device),target.to(device)
            data=data.view(-1,784)#------
            output=model(data)
            test_loss+=F.nll_loss(output,target,reduction='sum').item()
            pred=output.argmax(dim=1,keepdim=True)#row wise
            correct+=pred.eq(target.view_as(pred)).sum().item()
            
    #testing  
    test_loss/=len(test_loader.dataset)
    print("The loss is : ",test_loss)
    print("The success_rate is : ",100*correct/len(test_loader.dataset))

def main():

    global model #global variable for moded we set up for our neural network
    
    step_size=0.01 # step size or the learning rate
    
    batch_size=128     #setting the size of the batch which is 128
    
    #Data Loaders
    train_loader=torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST/processed',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_size,shuffle=True)

    test_loader=torch.utils.data.DataLoader(datasets.MNIST('./data/MNIST/processed',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=10000,shuffle=True)

    #Optimiser to optime our Neural Networks
    optimiser=optim.SGD(model.parameters(),lr=step_size,momentum=0.9,weight_decay=1e-3)
    
    #Number of epochs as in one epoch whole training dataset is used for training in batches in the batch size here of 128
    epochs=12
    
    t3=0#Reference variable used for calculating the time of training
    

    print("--------------------------Training and Testing using Neural Network------------------  :")

    for epoch in range(1,epochs+1):
        t0=time()
        train(model=model,train_loader=train_loader,optimiser=optimiser,loss='CE',epoch=epoch)
        t1=time()
        test(model=model,test_loader=test_loader)
        t3+=(t1-t0)
    
    #Total Time for training
    print("Total Time takan to train finally at such accuracy : ",t3)

if __name__=="__main__":
    main()