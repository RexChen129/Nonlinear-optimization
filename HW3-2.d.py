import torch
import matplotlib.pyplot as plt
x=torch.rand(50,10)-1/2
p=torch.ones(50,1)/2
y1=torch.bernoulli(p)
idxy=y1.eq(0)
y1[idxy]=-1
w_star=torch.ones(10,1)
y2=torch.sign(torch.matmul(x,w_star))

w=torch.randn(10,1)

lambda_max=torch.max(torch.eig(torch.matmul(torch.transpose(x,0,1),x))[0])

L=1/200*lambda_max
alpha=1/L
x=x.resize(50,10)
iteration_time = 1000
grad_norm=[]

for iter in range(iteration_time):
    f = 0
    g = 0
    for j in range(50):
        tmp=torch.exp(-y1[j] * torch.matmul(torch.transpose(w,0,1), x[j]))
        f = torch.log(1 + tmp) +f
        g = (-y1[j]*x[j]*tmp)/(1+tmp)+g
    f=f/50
    g=g/50
    g = g.view(-1,1)
    grad_norm.append(torch.log10(torch.norm(g)))
    print("Loss: %f; ||g||: %f" % (f, torch.norm(g)))
    w = w - alpha*g
print(w)


itr_time = range(0, iteration_time)

#plt.plot(itr_time, func_list1,label='x=(2,3)')
plt.plot(itr_time, grad_norm,label='Separable Case')
plt.title('Gradient norm v.s. iteration in 1000 iterations')
plt.ylabel('Log of gradient norm')
plt.legend()

plt.show()