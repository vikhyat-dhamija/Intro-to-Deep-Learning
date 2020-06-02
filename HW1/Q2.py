import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math


# load mini training data and labels
mini_train = np.load('knn_minitrain.npy')
mini_train_label = np.load('knn_minitrain_label.npy')

# randomly generate test data
mini_test = np.random.randint(20, size=20)
mini_test = mini_test.reshape(10,2)


# Define knn classifier
def kNNClassify(newInput, dataSet, labels, k):
        #array for storing the resulting labels of the testing data that is the random points being generated in the program
        result=[]
    ########################
    # Input your code here #
    ########################
        x=0
        #x is a counter variable for the loop for the number of testing data which is 10 in our program
        while (x < 10 ):

            np_distance=np.array([])
            #np_distance is an empty np array for distances and their corresponding labels
            for y in range(40):
                sum1=0
                sub=np.subtract(newInput[x],dataSet[y])#first subtract the testing point from the training point
                square=np.power(sub,2)#squaring the diffrence
                sum1=np.sum(square)# here summing to calculate x^2 + y^2 
                np_distance=np.append(np_distance,np.array([math.sqrt(sum1)]), axis=0)# storing the distance 
                np_distance=np.append(np_distance,np.array([labels[y]]),axis=0)#storing the label
            #reshaping the array to 40 into 2 two d array    
            np_distance= np_distance.reshape(40,2)           
            np_distance = np_distance[np_distance[:,0].argsort(kind='mergesort')]
            #sorting distances 
                    
            j=0
            #As there are four labels assigned
            labels_count=[0,0,0,0]
            #coverting the column of numpy array into the int as float is their default datatype
            label=np_distance[:,1]
            label=label.astype('int64')
            
            

            #counting the labels of the k smallest or nearest distance training points
            while(j < k):
                labels_count[label[j]]+=1
                j+=1
            
            j=0
            max=labels_count[0]
            #finding the maximum out of them
            max_index=0
            while(j < 4):
                if(labels_count[j] > max ):
                    max_index=j
                    max=labels_count[j]
                j+=1     
            #storing the classified label in the result
            result.append(max_index)
            x+=1
               
    ####################
    # End of your code #
    ####################
        return result

outputlabels=kNNClassify(mini_test,mini_train,mini_train_label,3)

print ('random test points are:', mini_test)
print ('knn classfied labels for test:', outputlabels)

# plot train data and classfied test data
train_x = mini_train[:,0]
train_y = mini_train[:,1]
fig = plt.figure()
plt.scatter(train_x[np.where(mini_train_label==0)], train_y[np.where(mini_train_label==0)], color='red')
plt.scatter(train_x[np.where(mini_train_label==1)], train_y[np.where(mini_train_label==1)], color='blue')
plt.scatter(train_x[np.where(mini_train_label==2)], train_y[np.where(mini_train_label==2)], color='yellow')
plt.scatter(train_x[np.where(mini_train_label==3)], train_y[np.where(mini_train_label==3)], color='black')

test_x = mini_test[:,0]
test_y = mini_test[:,1]
outputlabels = np.array(outputlabels)
plt.scatter(test_x[np.where(outputlabels==0)], test_y[np.where(outputlabels==0)], marker='^', color='red')
plt.scatter(test_x[np.where(outputlabels==1)], test_y[np.where(outputlabels==1)], marker='^', color='blue')
plt.scatter(test_x[np.where(outputlabels==2)], test_y[np.where(outputlabels==2)], marker='^', color='yellow')
plt.scatter(test_x[np.where(outputlabels==3)], test_y[np.where(outputlabels==3)], marker='^', color='black')

#save diagram as png file
plt.savefig("miniknn.png")
