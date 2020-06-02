import math
import numpy as np
import dask

#Mode is a variable whose value can be 0 or 1 : 1 for normal operation and 0 for differential operation

#inputs to the graph
x1=input("Please enter the value of x1 ")
x2=input("Please enter the value of x2 ")
w1=input("Please enter the value of w1 ")
w2=input("Please enter the value of w2 ")


#Forward Propagation in the graph


# Functions to be used
#Node 1 Multiplication
def node1(a,b,mode,pos,b_input):
    if mode==1:
        return a*b
    elif pos==1:
        return (b*b_input)
    else:
        return (a*b_input)
        

#Node 2 Cosine
def node2(a,mode,b_input):
    
    if mode==1:
        return (math.cos(a))
    else:
        return (-(math.sin(a))*b_input)

#Node 3 Multiplication2
def node3(a,b,mode,pos,b_input):
    if mode==1:
        return a*b
    elif pos==1:
        return (b*b_input)
    else:
        return (a*b_input)

#Node 4 sine function
def node4(a,mode,b_input):
    
    if mode==1:
        return math.sin(a)
    else:
        return ((math.cos(a))*b_input)

#Node 5 square function
def node5(a,mode,b_input):
    if mode==1:
        return a*a  
    else:
        return 2*a*b_input

#Node 6 addition1 function
def node6(a,b,mode,b_input):
    if mode==1:
        return a+b
    else:
        return (1*b_input)

#Node7 addition2 function
def node7(a,b,mode,b_input):
    if mode==1:
        return a+b
    else:
        return (1*b_input)

#Node 8 reciprocal function
def node8(a,mode,b_input):
    if mode==1:
        return (1/a)
    else:
        return(-1/(a*a))


#Forward Computation
input1_n3=float(x1)
input2_n3=float(w1)

input1_n1=float(x2)
input2_n1=float(w2)

print("Forward Propagation.........")

#Edge n3-n4-n5
print("Forward outputs at Edge n3-n4-n5")

output_n3=node3(input1_n3,input2_n3,1,0,0)
print("The output at Node 3",output_n3)

input_n4=output_n3
output_n4=node4(input_n4,1,0)
print("The output at Node 4",output_n4)


input_n5=output_n4
output_n5=node5(input_n5,1,0)
print("The output at Node 5",output_n5)

#Edge n1-n2
print("Forward outputs at Edge n1-n2")
output_n1=node1(input1_n1,input2_n1,1,0,0)
print("The output at Node 1",output_n1)

input_n2=output_n1
output_n2=node2(input_n2,1,0)
print("The output at Node 2",output_n2)

print("Forward outputs at Edge n6-n7-n8")
input1_n6=output_n5
input2_n6=output_n2
output_n6=node6(input1_n6,input2_n6,1,0)
print("The output at Node 6",output_n6)

input_n7=output_n6
output_n7=node7(2,input_n7,1,0)
print("The output at Node 7",output_n7)

input_n8=output_n7
output_n8=node8(input_n8,1,0)
print("The output at Node 8",output_n8)

print("The Final Result of the Forward Computation is : ",output_n8 )

#Backward Propagation
print("Backward Propagation.........")
print("Backward differential outputs at Edge n8-n7-n6")

b_output_n8=node8(input_n8,0,1)
print("Backward differential output at Node 8 : ",b_output_n8)

b_output_n7=node7(2,input_n7,0,b_output_n8)
print("Backward differential output at Node 7 : ",b_output_n7)

b_output_n6=node6(input1_n6,input2_n6,0,b_output_n7)
print("Backward differential output at Node 6 : ",b_output_n6)

print("Backward differential outputs at Edge n5-n4-n3")
b_output_n5=node5(input_n5,0,b_output_n6)
print("Backward differential output at Node 5 : ",b_output_n5)

b_output_n4=node4(input_n4,0,b_output_n5)
print("Backward differential output at Node 4 : ",b_output_n4)

b_output_n3_x1=node3(input1_n3,input2_n3,0,1,b_output_n4)
b_output_n3_w1=node3(input1_n3,input2_n3,0,2,b_output_n4)
print("Backward differential output at Node 3 position 1 output with respect to x1: ",b_output_n3_x1)
print("Backward differential output at Node 3 position 2 output with respect to w1: ",b_output_n3_w1)


print("Backward differential outputs at Edge n2-n1")
b_output_n2=node2(input_n2,0,b_output_n6)
print("Backward differential output at Node 2 : ",b_output_n2)

b_output_n1_x2=node1(input1_n1,input2_n1,0,1,b_output_n2)
b_output_n1_w2=node1(input1_n1,input2_n1,0,2,b_output_n2)
print("Backward differential output at Node 1 position 1 output with respect to x2: ",b_output_n1_x2)
print("Backward differential output at Node 1 position 2 output with respect to w2: ",b_output_n1_w2)

print("Overall output..........")

print("The Result of the partial differentiation of output with respect to x1  is : ",b_output_n3_x1)

print("The Result of the partial differentiation of output with respect to w1  is : ",b_output_n3_w1)

print("The Result of the partial differentiation of output with respect to x2  is : ",b_output_n1_x2)

print("The Result of the partial differentiation of output with respect to w2  is : ",b_output_n1_w2)
