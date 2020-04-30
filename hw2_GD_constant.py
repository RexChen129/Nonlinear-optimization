import torch
import matplotlib.pyplot as plt
a1=torch.rand(100,50)
a2=torch.rand(100, 50)*5
A=torch.cat((a1,a2),1)
alpha = 0.00002
iteration_time=1000
X=torch.ones(100,1)

Loss_list = []
Grad_norm = []
for iter in range(iteration_time):

    f = torch.pow(torch.norm(torch.matmul(torch.transpose(A,0,1),X)),2)
    g = torch.matmul(torch.matmul(A,torch.transpose(A,0,1)),X)*2

    print("Loss: %f; ||g||: %f,iteration: %f" % (f, torch.norm(g),iter))
    g = g.view(-1,1)
    X = X - alpha*g
    Loss_list.append(torch.log(f))
    Grad_norm.append(torch.log(torch.norm(g)))
print(X)

itr_time = range(0, iteration_time)
plt.subplot(1, 2, 1)
plt.plot(itr_time, Loss_list,label='GD with constant stepsize')
plt.title('Log of Objective Value v.s. iteration')
plt.ylabel('Log of Objective value')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(itr_time, Grad_norm,label='GD with constant stepsize')
plt.title('Log of Gradient Norm v.s. iteration')
plt.ylabel('Log of Gradient Norm')
plt.legend()
plt.show()