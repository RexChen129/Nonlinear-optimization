import torch
import matplotlib.pyplot as plt
a=torch.tensor([3,2,-0.1])
Q=torch.diag(a)

alpha = 0.6


def count(iteration_time,X):
    function_error=[]
    for iter in range(iteration_time):
        f = torch.matmul(torch.matmul(torch.transpose(X,0,1),Q),X)*0.5
        g = torch.matmul(Q,X)
        #print("Loss: %f; ||g||: %f,iteration: %f, error:%f" % (torch.log10(f), torch.norm(g),iter,torch.log10(itr_error*itr_error)))
        g = g.view(-1,1)
        X = X - alpha*g
        #Loss_list.append(torch.log(f))
        #Grad_norm.append(torch.log(torch.norm(g)))
        function_error.append(torch.log10(f))
    return(function_error)

f1=count(1000,torch.ones(3,1))
#f2=count(1000000,torch.ones(3,1))
itr_time1 = range(0, 1000)
#itr_time2 = range(0, 10000)
plt.subplot(1, 2, 1)
plt.plot(itr_time1, f1,label='GD with constant stepsize')
plt.title('Function_error v.s. iteration in 1000 iterations')
plt.ylabel('Log of function_error')
plt.legend()
#plt.subplot(1, 2, 2)
#plt.plot(itr_time2, f2,label='GD with constant stepsize')
#plt.title('Function_error v.s. iteration in 1000000 iterations')
#plt.ylabel('Log of function_error')
#plt.legend()
plt.show()