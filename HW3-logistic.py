import torch
import matplotlib.pyplot as plt

y = torch.Tensor([1,-1])

iteration_time = 10000
def count(X):
    w = torch.Tensor([2])
    func_list=[]
    grad_norm=[]
    alpha = 8 / (torch.pow(X[0], 2) + torch.pow(X[1], 2))
    for iter in range(iteration_time):
        f = 1/2*torch.log(1+torch.exp(-y[0]*w*X[0]))+1/2*torch.log(1+torch.exp(-y[1]*w*X[1]))
        g = 1/2*(-y[0]*X[0]*torch.exp(-y[0]*w*X[0]))/(1+torch.exp(-y[0]*w*X[0]))+\
            1/2*(-y[1]*X[1]*torch.exp(-y[1]*w*X[1]))/(1+torch.exp(-y[1]*w*X[1]))
        func_list.append(torch.log10(f))
        grad_norm.append(torch.log10(torch.norm(g)))
        print("Loss: %f; ||g||: %f" % (f, torch.norm(g)))
        g = g.view(-1,1)
        w = w - alpha*g
    print(w)
    return func_list,grad_norm

#func_list1,grad_norm1=count(torch.Tensor([2,3]))
#print(grad_norm1)
func_list2,grad_norm2=count(torch.Tensor([2,-3]))
itr_time = range(0, iteration_time)
plt.subplot(1, 2, 1)
#plt.plot(itr_time, func_list1,label='x=(2,3)')
plt.plot(itr_time, func_list2,label='x=(2,-3)')
plt.title('Function values v.s. iteration in 10000 iterations')
plt.ylabel('Log of function values')
plt.legend()
plt.subplot(1, 2, 2)
#plt.plot(itr_time, grad_norm1,label='x=(2,3)')
plt.plot(itr_time, grad_norm2,label='x=(2,-3)')
plt.title('Gradient norm v.s. iteration in 10000 iterations')
plt.ylabel('Log of gradient norm')
plt.legend()
plt.show()