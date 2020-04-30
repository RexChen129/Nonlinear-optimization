import numpy as np
from autograd import grad
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.linalg as lin
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# print(R)
# Z = np.sin(R)
#
#
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
#
# plt.show()

# GD for general problem
def GD( func , w0, stepsize = 1, MaxIter = 1000 ):
  iter_list = []
  loss_list = []
  w = w0
  grad_cur =  grad(func)
  for iter_count in range(MaxIter):
     w = w - stepsize*grad_cur(w)
     loss = func(w)
     iter_list.append ( iter_count )           # element-wise product of the weights w
     loss_list.append( loss.item(0) )    # vector, with each entry lambda_i w_i^2
  return iter_list, loss_list

# BB for general problem
def BB( func , w0, MaxIter = 1000 ):
  iter_list = []
  loss_list = []
  w = w0
  grad_cur =  grad(func)
  g = grad_cur(w)
  wnew = w - 1*g
  gnew =  grad_cur(wnew)
  s = wnew-w
  y = gnew-g
  for iter_count in range(MaxIter):
     loss = func(w)
     stepsize = (s.T@s) / (s.T@y)
     w = wnew
     g = gnew
     wnew = w - stepsize*g
     gnew = grad_cur(wnew)
     s = wnew-w
     y = gnew-g
     iter_list.append ( iter_count )           # element-wise product of the weights w
     loss_list.append( loss.item(0) )    # vector, with each entry lambda_i w_i^2
  return iter_list, loss_list

# BFGS for general problem
def getpath_cb(func, loss_list=[], iter_list=[], iter_count=0):
    def getpath(xk):
        nonlocal func
        nonlocal iter_count
        iter_count+=1
        iter_list.append(iter_count)
        loss_list.append(func(xk))
    return getpath

def func_grad(x):
  temp_grad = A @ x
  return temp_grad

# Define BFGS method. Check the doc at https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html
# Need two extra things: 1) func_grad to provide the gradient of the function. Here We use an explicit form; could use autograd to define grad(func)
#  2) getpath_cb to record the intermediate iterates
def BFGS(func, w0, MaxIter=1000):
  iter_count=0
  loss_list = []
  iter_list = []
  op = { 'gtol': 1e-20, 'maxiter': MaxIter}
  res = opt.minimize(func, w0, method='BFGS', jac = func_grad, options= op, callback=getpath_cb(func, loss_list, iter_list, iter_count))
  return iter_list, loss_list

# Define L-BFGS method. Check the doc at https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
def LBFGS(func, w0, MaxIter=1000):
  iter_count=0
  loss_list = []
  iter_list = []
  op = {'maxcor': 10, 'ftol': 1e-15, 'gtol': 1e-20, 'maxfun':15000, 'maxiter': MaxIter, 'maxls': 10}
  res = opt.minimize(func, w0, method='L-BFGS-B', jac = func_grad, options=op, callback=getpath_cb(func, loss_list, iter_list, iter_count))
  return iter_list, loss_list


# Heavy Ball for general problem
# Set default beta to be 0.9; conservative is better than agressive
def HB( func , w0, stepsize = 1, momentum = 0.9, MaxIter = 1000 ):
  iter_list = []
  loss_list = []
  w = w0
  grad_cur =  grad(func)
  wnew = w - stepsize*grad_cur(w)
  s = wnew-w
  for iter_count in range(MaxIter):
     w=wnew
     wnew = w - stepsize*grad_cur(w)+momentum*s
     s = wnew-w
     loss = func(w)
     iter_list.append ( iter_count )           # element-wise product of the weights w
     loss_list.append( loss.item(0) )    # vector, with each entry lambda_i w_i^2
  return iter_list, loss_list

## Diagonal Matrix Definitions

# Toy Case: Simplest low-dim diagonal matrix
# d1 = 3
#A = np.eye(d)
# D1 = np.diag([1, 1e-2, 1e-4])

# Case 1 (3 clusters): Medium-dim diagonal matrix, with 3 clusters
d1 = 300
v = np.ones( [1, d1] )
for j in range( np.int(d1/3 ) ):
   v[0, j] = 1e-2
for j in range( np.int(d1/3), np.int(d1*2/3) ):
   v[0, j] = 1e-4
D1 = np.diag( v[0] )

# Case 2 (linear range): Medium-dim diagonal matrix, with linear range
d2 = 1000
spacing = 10
cons = 0
v =  np.arange( 1, d2*spacing, spacing  ) + cons    # between 1/d to 1, equal space
v = v/max(v)
D2 = np.diag( v )

# Case 3 (few large plus linear range): Linear range plus large few eigenvalues
# Case 3 (few large plus linear range): Linear range plus large few eigenvalues
d3 = 100
spacing = 10
cons = 0
intended_max = d3*spacing
v =  ( np.arange( 1, d3*spacing, spacing  ) + cons ) # / intended_max    # between 1/d to 1, equal space
v[d3-1] = 10*d3*spacing
v[d3-2] = 6*d3*spacing
v[d3-3] = 5*d3*spacing
v = v/max(v)
D3 = np.diag( v )
a,b=np.linalg.eig(D3)


# Case 4: control condition number to be a pre-defined value kappa. Random eigenvalues in between
d4 = 500
kappa = 150
sigma = (np.random.rand(d4, 1)+1/np.sqrt(kappa))*10
sigma[0] = 10
sigma[d4 -1] = 10/np.sqrt(kappa)
sigma_fl = sigma.flatten()
D4 = np.diag(sigma_fl)
# print("D = ",  D )

def quadratic(A, x):
  y = 0.5 * x.T @ A @ x
  return y

def quadloss(w):
  lossquad = quadratic(A, w)
  return lossquad

# Define the matrix A and d
#============ Major change ===============
A = D3
d = np.size(A,0)
# print(A)

L = max(np.linalg.eigvals(A))
mu = min(np.linalg.eigvals(A))
print("L = ", L)
print("mu =", mu)

w0 = 5* np.ones([d,1]) # random.randn(3,1)
MaxIter_check = 10000

def search_stepsize(start,step):
    loss_tmp = 100
    n=0
    for i in np.arange(start,5,step):
        n+=1
        iter_list1, loss_list1 = GD(quadloss, w0, stepsize=i,MaxIter = MaxIter_check)
        loss=loss_list1[-1]
        if loss<loss_tmp:
            loss_tmp=loss
            continue
        else:
            st=i-step
            break
    return st
# loss=[]
# for i in np.arange(0.1,2.1,0.1):
#     iter_list1, loss_list1 = GD(quadloss, w0, stepsize=i, MaxIter=MaxIter_check)
#     loss.append(loss_list1[-1])
#
#
# plt.semilogy(np.arange(0.1,2.1,0.1), loss, 'r-', label = 'GD',  lw = 2)
# plt.legend()
# plt.xlabel("stepsize")
# plt.ylabel("loss")
# plt.title('Stepsize v.s. loss for d = %d' % d)
# plt.show()
# st1=search_stepsize(0.1,0.5)
# print(st1)
# st2=search_stepsize(st1,0.1)
# print(st2)
# st3=search_stepsize(st2,0.01)
# print(st3)
# st4=search_stepsize(st3,0.001)
# print(st4)

def search_hb_stepsize_alpha(start,step,beta):
    loss_tmp = 1000
    n=0
    for i in np.arange(start,5,step):
        n+=1
        print(n)
        iter_list5, loss_list5 = HB( quadloss, w0, stepsize=i, momentum = beta, MaxIter = MaxIter_check)
        loss=loss_list5[-1]
        if loss<loss_tmp:
            loss_tmp=loss
            continue
        else:
            st=i-step
            break
    return st
def search_hb_stepsize_beta(start,step,alpha):
    loss_tmp = 1000
    n=0
    for i in np.arange(start,5,step):
        n+=1
        print(n)
        iter_list5, loss_list5 = HB( quadloss, w0, stepsize=alpha, momentum = i, MaxIter = MaxIter_check)
        loss=loss_list5[-1]
        if loss<loss_tmp:
            loss_tmp=loss
            continue
        else:
            st=i-step
            break
    return st
b1=search_hb_stepsize_beta(0.1,0.5,1)
a1=search_hb_stepsize_alpha(0.1,0.5,b1)
print(b1,a1)
b2=search_hb_stepsize_beta(b1,0.1,a1)
a2=search_hb_stepsize_alpha(a1,0.1,b2)
print(b2,a2)
b3=search_hb_stepsize_beta(b2,0.01,a2)
a3=search_hb_stepsize_alpha(a2,0.01,b3)
print(b3,a3)
b4=search_hb_stepsize_beta(b3,0.001,a3)
a4=search_hb_stepsize_alpha(a3,0.001,b4)
print(b4,a4)
# fig = plt.figure()
# ax = Axes3D(fig)
# a_range=np.arange(0.1,2,0.1)
# b_range=np.arange(0.1,1,0.1)
# loss=[]
# for i in a_range:
#     for j in b_range:
#         iter_list5, loss_list5 = HB( quadloss, w0, stepsize=i, momentum = j, MaxIter = MaxIter_check)
#         loss.append(loss_list5[-1])
#
# a, b = np.meshgrid(a_range, b_range)
# loss=np.array(loss).reshape(len(b_range),len(a_range))
#
#
# ax.plot_surface(a, b, loss, rstride=1, cstride=1, cmap='rainbow')
#
# plt.show()
# st1=search_hb_stepsize_alpha(0.1,0.5)
# st2=search_hb_stepsize_alpha(st1,0.1)
# st3=search_hb_stepsize_alpha(st2,0.01)
# st4=search_hb_stepsize_alpha(st3,0.001)
# print(st4)
#print(loss_list1)
#iter_list2, loss_list2 = BB( quadloss, w0, MaxIter = MaxIter_check)
#print(loss_list2)
# iter_list3, loss_list3 = BFGS( quadloss, w0, MaxIter = MaxIter_check)
# #print(loss_list3)
# iter_list4, loss_list4 = LBFGS( quadloss, w0, MaxIter = MaxIter_check)
# #print(loss_list4)


# plt.semilogy(iter_list1, loss_list1, 'r-', label = 'GD',  lw = 2)
#
# #plt.semilogy(iter_list2, loss_list2, 'g--', label = 'BB',  lw = 2)
#
# # plt.semilogy(iter_list3, loss_list3, 'b--', label = 'BFGS',  lw = 2)
# #
# # plt.semilogy(iter_list4, loss_list4, 'y--', label = 'L-BFGS',  lw = 2)
# #
# # plt.semilogy(iter_list5, loss_list5, 'm', label = 'HB',  lw = 2)
#
# plt.legend()
# plt.xlabel("iter")
# plt.ylabel("loss")
#
# plt.ylim(1e-6,1e3)
#
# plt.title('compare BB, BFGS and GD for d = %d' % d)
# plt.show()

# # Consider different A's
# d= 500
# n= 600
#
# # Random Gaussian
# mu = 0
# A1 = np.random.randn(n, d)-mu
# A1 = A1.T @ A1
# #print(A1)
# #print(np.linalg.eig(A1))
#
# # Random uniform
# mu = 0
# A2 = np.random.rand(n, d)- mu
# A2 = A2.T @ A2
# #print(A2)
# #print(np.linalg.eig(A2))
#
# # Controlled spectrum +  random Gaussian matrices
# D = np.sqrt(D3)
# d =  D.shape[0]
# print(d)
# U = np.random.randn(d,d)
# V = np.random.randn(d,d)
# A3 = lin.orth(U) @ D @ lin.orth(V)
# A3 = A3.T @ A3
# #print(A3)
# # print("eigenvalues are \n", np.linalg.eig(A3))
#
# # Controlled spectrum  +  random uniform matrices
# D = D3  # using the definition of D in the previous block
# d =  D.shape[0]
# U = np.random.rand(d,d)
# V = np.random.rand(d,d)
# A4 = lin.orth(U) @ D @lin.orth(V)
# A4 = A4.T @ A4
# # print("eigenvalues are \n", np.linalg.eigvals(A4))
# #print(A4)
#
# # Remark: Should use n*d matrix to generate Q = A'A, which is closer to practice.
#
# A = A3  # Adjust the coefficient matrix
# d = A.shape[0]
# print(A.shape[0])
#
# L = max(np.linalg.eigvals(A))
# mu = min(np.linalg.eigvals(A))
# print("L = ", L)
# print("mu =", mu)
#
# w0 = 5* np.ones([d,1]) # random.randn(3,1)
# L = max(np.linalg.eigvals(A))
# mu = min(np.linalg.eigvals(A))
# #print(L)
# #print(mu)
# MaxIter_check = 5000
# iter_list1, loss_list1 = GD( quadloss, w0, stepsize=1/L, MaxIter = MaxIter_check)
# #print(loss_list1)
# iter_list2, loss_list2 = BB( quadloss, w0, MaxIter = MaxIter_check)
# #print(loss_list2)
# iter_list3, loss_list3 = BFGS( quadloss, w0, MaxIter = MaxIter_check)
# #print(loss_list3)
# iter_list4, loss_list4 = LBFGS( quadloss, w0, MaxIter = MaxIter_check)
# #print(loss_list4)
# iter_list5, loss_list5 = HB( quadloss, w0, stepsize=1/L,  MaxIter = MaxIter_check) # momentum=(1-np.sqrt(mu/L))**2,
# # Remark: For HB, we could use the optimal parameter predicted by theory, but use it with caution.
# #print(loss_list5)
#
# plt.semilogy(iter_list1, loss_list1, 'r-', label = 'GD',  lw = 2)
# plt.semilogy(iter_list2, loss_list2, 'g--', label = 'BB',  lw = 2)
# plt.semilogy(iter_list3, loss_list3, 'b--', label = 'BFGS',  lw = 2)
# plt.semilogy(iter_list4, loss_list4, 'y-', label = 'L-BFGS',  lw = 2)
# plt.semilogy(iter_list5, loss_list5, 'm', label = 'HB',  lw = 2)
#
# plt.legend()
# plt.xlabel("iter")
# plt.ylabel("loss")
#
# plt.ylim( 1e-12, 1e8)
#
# # plt.title('compare BB, BFGS and GD')
# # plt.title(' d= %d , n = % d, condition number = %d' % (d, n, kappa))
# plt.title(' d= %d , n = % d' % (d, n))
# plt.show()