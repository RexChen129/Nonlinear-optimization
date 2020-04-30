import torch
import matplotlib.pyplot as plt
Q=torch.tensor([[1,0],[0,1e-6]])

alpha = 1
iteration_time=1000001
X=torch.ones(2,1)
x_opt=torch.zeros(2,1)
Loss_list = []
Grad_norm = []
iteration_error= []
function_error= []
for iter in range(iteration_time):

    f = torch.matmul(torch.matmul(torch.transpose(X,0,1),Q),X)*0.5
    g = torch.matmul(Q,X)
    itr_error=torch.norm(X)
    fc_error=torch.norm(f-0)
    print("Loss: %f; ||g||: %f,iteration: %f, error:%f" % (torch.log10(fc_error), torch.norm(g),iter,torch.log10(itr_error*itr_error)))
    g = g.view(-1,1)
    X = X - alpha*g
    #Loss_list.append(torch.log(f))
    #Grad_norm.append(torch.log(torch.norm(g)))
    iteration_error.append(2*torch.log10(itr_error))
    function_error.append(torch.log10(fc_error))

print(X)

itr_time = range(0, iteration_time)
plt.subplot(1, 2, 1)
plt.plot(itr_time, iteration_error,label='GD with constant stepsize')
plt.title('Iteration_error v.s. iteration in 10000 iterations')
plt.ylabel('Log of iteration_error')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(itr_time, function_error,label='GD with constant stepsize')
plt.title('Function_error v.s. iteration in 10000 iterations')
plt.ylabel('Log of function_error')
plt.legend()
plt.show()