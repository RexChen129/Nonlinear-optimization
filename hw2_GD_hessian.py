import torch
import matplotlib.pyplot as plt
torch.manual_seed(7)
a1=torch.rand(100,50)
a2=torch.rand(100, 50)*5
A=torch.cat((a1,a2),1)
Aj_norm = torch.reciprocal(torch.norm(A,p=2,dim=0), out=None)
A_tuta =torch.mul(A, Aj_norm, out=None)
#alpha = 0.01
iteration_time=50000
u=torch.sum(A_tuta, 0, out=None)/100
B=torch.add(A_tuta, -u, out=None)
X=torch.randn(100,1)
X1=torch.randn(100,1)

Loss_list = []
Grad_norm = []
Loss_list1 = []
Grad_norm1 = []
E,V=torch.eig(torch.matmul(A,torch.transpose(A,0,1))*2)
E1,V1=torch.eig(torch.matmul(A_tuta,torch.transpose(A_tuta,0,1))*2)

alpha=1/torch.max(E)
alpha1=1/torch.max(E1)
for iter in range(iteration_time):

    f = torch.pow(torch.norm(torch.matmul(torch.transpose(A,0,1),X)),2)
    g = torch.matmul(torch.matmul(A,torch.transpose(A,0,1)),X)*2
    print("Loss: %f; ||g||: %f,iteration: %f" % (f, torch.norm(g),iter))
    g = g.view(-1,1)
    X = X - alpha*g
    Loss_list.append(torch.log(f))
    Grad_norm.append(torch.log(torch.norm(g)))


for iter in range(iteration_time):

    f = torch.pow(torch.norm(torch.matmul(torch.transpose(A_tuta,0,1),X1)),2)
    g = torch.matmul(torch.matmul(A_tuta,torch.transpose(A_tuta,0,1)),X1)*2

    print("Loss: %f; ||g||: %f,iteration: %f" % (f, torch.norm(g),iter))
    g = g.view(-1,1)
    X1 = X1 - alpha1*g
    Loss_list1.append(torch.log(f))
    Grad_norm1.append(torch.log(torch.norm(g)))
print(torch.max(E))
print(torch.max(E1))

itr_time = range(0, iteration_time)
plt.subplot(1, 2, 1)
plt.plot(itr_time, Loss_list,label='GD with A')
plt.plot(itr_time, Loss_list1,label='GD with $\widetilde{ A }$')
plt.title('Log of Objective Value v.s. iteration in Hessians')
plt.ylabel('Log of Objective value')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(itr_time, Grad_norm,label='GD with A')
plt.plot(itr_time, Grad_norm1,label='GD with $\widetilde{ A }$')
plt.title('Log of Gradient Norm v.s. iteration  in Hessians')
plt.ylabel('Log of Gradient Norm')
plt.legend()
plt.show()