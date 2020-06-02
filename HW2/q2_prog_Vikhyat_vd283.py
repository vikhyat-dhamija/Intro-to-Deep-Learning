import math
import numpy as np

w=[0]*9
x=[0]*3
print("Please enter W the  3 * 3 matrix....")
for i in range(9):
    w[i]=float(input())


print("Please enter X the  3 * 1 matrix....")
for m in range(3):
    x[m]=float(input())  
        

#Mode is a variable whose value can be 0 or 1 : 1 for normal operation and 0 for differential operation

#Node 1 Matrix Multiplication or Dot Product
def node1(a,b,mode,pos,b_input):
    if mode==1:
       return np.dot(a,b)
    elif pos==1:
       return np.dot(b_input,np.transpose(b))
    else:
        return np.dot(np.transpose(a),b_input)

#Node 2 Sigmoid Function
def node2(a,mode,b_input):
    if mode==1:
       return (1/(1 + np.exp(-a)))
    else:
       return np.multiply((np.exp(-a)/np.square(1 + np.exp(-a))),b_input)

#Node 3 Summation Function
def node3(a,mode,b_input):
    if mode==1:
        return(np.sum(np.square(a)))
    else:
        return(a*2*b_input)

#Forward Propagation
print("Forward propagation........","\n")
#input1_n1=np.array([[1,0,1],[1,0,0],[0,1,1]])
#input2_n1=np.array([[1],[0],[1]])
input1_n1=np.array(w).reshape([3,3])
input2_n1=np.array(x).reshape([3,1])
#Computation of Node 1
output_n1=node1(input1_n1,input2_n1,1,0,0)
print("The forward path output at Node 1 : ","\n",output_n1)

#Computation of Node 2
input_n2=output_n1
output_n2=node2(input_n2,1,0)
print("The forward path output at Node 2 : ","\n",output_n2)

#Computation of Node 3
input_n3=output_n2
output_n3=node3(input_n3,1,0)
print("The forward path output at Node 3 : ","\n",output_n3)

print("The Result of the Forward Computation is : ","\n",output_n3)

#Backward Propagation
print("Backward Propagation..........")
#Node 3
b_output_n3=node3(input_n3,0,1)
print("The back ward differentiation output at Node 3 is : ","\n",b_output_n3)

#Node 2
b_output_n2=node2(input_n2,0,b_output_n3)
print("The back ward differentiation output at Node 2 is : ","\n",b_output_n2)
#Node 1

#Partial Differentiation with respect to w
b_output_n1_w=node1(input1_n1,input2_n1,0,1,b_output_n2)
print("The back ward differentiation output at Node 1 at position 1 result partial diff of output w.r.t W is : ","\n",b_output_n1_w)

#Partial Differentiation with respect to x
b_output_n1_x=node1(input1_n1,input2_n1,0,2,b_output_n2)
print("The back ward differentiation output at Node 1 at position 2 result partial diff of output w.r.t x is : ","\n",b_output_n1_x)