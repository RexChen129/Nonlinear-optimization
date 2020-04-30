import torch
import matplotlib.pyplot as plt
torch.manual_seed(1)
a1=torch.rand(100,50)
a2=torch.rand(100, 50)*5
A=torch.cat((a1,a2),1)
Aj_norm = torch.reciprocal(torch.norm(A,p=2,dim=0), out=None)
A_tuta =torch.mul(A, Aj_norm, out=None)
#alpha = 0.01
iteration_time=3000
u=torch.sum(A_tuta, 0, out=None)/100
B=torch.add(A_tuta, -u, out=None)

E,V=torch.eig(torch.matmul(A,torch.transpose(A,0,1))*2)
E1,V1=torch.eig(torch.matmul(A_tuta,torch.transpose(A_tuta,0,1))*2)
E2,V2=torch.eig(torch.matmul(A_tuta,torch.transpose(B,0,1))*2)
H=torch.matmul(A,torch.transpose(A,0,1))*2
H1=torch.matmul(A_tuta,torch.transpose(A,0,1))*2
H2=torch.matmul(B,torch.transpose(A,0,1))*2
print(torch.norm(H)*torch.norm(torch.inverse(H)))
print(torch.norm(H1)*torch.norm(torch.inverse(H1)))
print(torch.norm(H2)*torch.norm(torch.inverse(H2)))
print(torch.max(E[:,0])/torch.min(E[:,0]))
print(torch.max(E1[:,0])/torch.min(E1[:,0]))
print(torch.max(E2[:,0])/torch.min(E2[:,0]))