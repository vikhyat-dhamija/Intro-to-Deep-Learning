import math
import numpy as np  
from download_mnist import load
import operator  
import time
# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)

#Function for Knn classification

def kNNClassify(newInput, dataSet, labels, k): 
        # variable for counting the length of input array of image matrices
        x=0
        # variable for storing the resulting labels for all the testing images
        result_labels=[]
        #upper loop for the number of testing images that is the length of the input array for testing
        while(x < len(newInput)):
            #empty array for storing the distances of the one testing image with 60k training images
            np_distance=np.array([])
            #loop for calculating the distances of the image being tested with the training images
            for y in range(60000):
                sum1=0
                sub=np.subtract(newInput[x],dataSet[y])#first subtraction
                square=np.power(sub,2)#then squaring
                sum1=np.sum(square)#then sum of all the values of the matrices
                #np_distance is having tuples with two entries 1 distance from training image and the corresponding label of training image
                np_distance=np.append(np_distance,np.array([math.sqrt(sum1)]), axis=0)
                np_distance=np.append(np_distance,np.array([labels[y]]),axis=0)
            #reshaping into two Dimension array   
            np_distance= np_distance.reshape(60000,2)           
            np_distance = np_distance[np_distance[:,0].argsort(kind='mergesort')]
            #Then sorting with respect to distance so that lowest distances come up           
            j=0
            #labels counting as labels are from 0 to 9 for digits
            labels_count=[0,0,0,0,0,0,0,0,0,0]
            #converting the second column of the two d array and then converting into int as float is default for numpy array
            label=np_distance[:,1]
            label=label.astype('int64')
            

            #For k lowest distance values we count the respective number of labels
            while(j < k):
                labels_count[label[j]]+=1
                j+=1
            #Then finding the maximum value of the label count 
            j=0
            max=labels_count[0]
            max_index=0
            while(j < 10):
                if(labels_count[j] > max ):
                    max_index=j
                    max=labels_count[j]
                j+=1     
            #then that label will be asigned in the result label vector matrices for the corresponding testing image
            result_labels.append(max_index)
            x+=1
            #incrementing value of x 
        return result_labels

start_time = time.time()
outputlabels=kNNClassify(x_test[0:10],x_train,y_train,10)
#printing the true labels of the testing images
print(y_test[0:10])
#printing the result after classification
print(outputlabels)
#resulting accuracy
result = y_test[0:10] - outputlabels
result = (1 - np.count_nonzero(result)/len(result))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))
