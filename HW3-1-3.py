import matplotlib.pyplot as plt
import math

k_list=[1,20,100,2000]
y2_list=[]
y3_list=[]
y4_list=[]
for i in k_list:
    y2=1e-2*pow(0.99,2*i)
    y3 = 1e-5 * pow(0.99999, 2 * i)
    y4=y2+y3
    y2=math.log10(y2)
    y3 = math.log10(y3)
    y4=math.log10(y4)
    y2_list.append(y2)
    y3_list.append(y3)
    y4_list.append(y4)
print(y2_list,"y2")
print(y3_list,"y3")
print(y4_list,"y4")



plt.plot(k_list, y2_list,label='log(Y2)')
plt.plot(k_list, y3_list,label='log(Y3)')
plt.plot(k_list, y4_list,label='log(Y2+Y3)')
#plt.title('Function_error v.s. iteration in 1000 iterations')
#plt.ylabel('Log of function_error')
plt.legend()
plt.show()