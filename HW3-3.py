import torch
import matplotlib.pyplot as plt
a=torch.tensor([3,2,0.01])
Q=torch.diag(a)
lambda_max=torch.max(torch.eig(Q)[0])
alpha = 1/lambda_max
x=torch.ones(3,1)
iteration_time=1000
beta=0.95
a1=(1+beta)*(torch.eye(3)-alpha*Q)
a2=beta*(alpha*Q-torch.eye(3))
aa=torch.cat((a1,a2),1)
bb=torch.cat((torch.eye(3),torch.diag(torch.zeros(3))),1)
M=torch.cat((aa,bb),0)
torch.eig(M)[0]
z=torch.ones(6,1)

function_error=[]
for iter in range(iteration_time):
    f = torch.matmul(torch.matmul(torch.transpose(x,0,1),Q),x)*0.5
    z = torch.matmul(M,z)
    g=torch.matmul(Q,x)
    #print("Loss: %f; ||g||: %f,iteration: %f, error:%f" % (torch.log10(f), torch.norm(g),iter,torch.log10(itr_error*itr_error)))
    x = z[0:3]
    #Loss_list.append(torch.log(f))
    #Grad_norm.append(torch.log(torch.norm(g)))
    function_error.append(torch.log10(torch.norm(g)))



itr_time1 = range(0, iteration_time)
itr_time2 = range(0, 10000)

plt.plot(itr_time1, function_error,label='beta=0.8985')
plt.title('Gradient norm v.s. iteration in 1000 iterations')
plt.ylabel('Log of Gradient')
plt.legend()
#plt.subplot(1, 2, 2)
#plt.plot(itr_time2, f2,label='GD with constant stepsize')
#plt.title('Function_error v.s. iteration in 1000000 iterations')
#plt.ylabel('Log of function_error')
#plt.legend()
plt.show()